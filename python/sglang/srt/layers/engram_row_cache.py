"""A set-associative host-RAM cache of Engram rows (weight bytes + scale bytes).

Row ids are n-gram hashes, so ``id % n_sets`` spreads them evenly; eight
least-recently-used ways per set track exact LRU closely at a fraction of the
memory an exact LRU map over ~19M rows would need (DSV41_REFERENCE §5).
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
from typing import Callable, Optional

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# One lookup per Engram layer per forward, so this logs about every 256 forwards.
LOG_EVERY_LOOKUPS = 512


class EngramRowCache:
    def __init__(
        self,
        capacity_rows: int,
        row_bytes: int,
        ways: int = 8,
        log_every: int = LOG_EVERY_LOOKUPS,
    ) -> None:
        self.ways = ways
        self.log_every = log_every
        self.n_sets = max(1, capacity_rows // ways)
        self.row_bytes = row_bytes
        self.tags = np.full((self.n_sets, ways), -1, dtype=np.int64)
        self.ages = np.zeros((self.n_sets, ways), dtype=np.int64)
        self.data = np.zeros((self.n_sets * ways, row_bytes), dtype=np.uint8)
        self.clock = 0
        self.accesses = 0
        self.hits = 0

    @classmethod
    def for_bytes(cls, budget_bytes: int, row_bytes: int) -> EngramRowCache:
        return cls(budget_bytes // row_bytes, row_bytes)

    def lookup(self, keys: np.ndarray, fetch: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        self.clock += 1
        self.accesses += keys.size
        unique, inverse = np.unique(keys, return_inverse=True)
        sets = unique % self.n_sets
        match = self.tags[sets] == unique[:, None]
        hit = match.any(axis=1)
        way = match.argmax(axis=1)
        out = np.empty((unique.size, self.row_bytes), dtype=np.uint8)
        out[hit] = self.data[sets[hit] * self.ways + way[hit]]
        self.ages[sets[hit], way[hit]] = self.clock
        miss = ~hit
        if miss.any():
            rows = fetch(unique[miss])
            out[miss] = rows
            for key, s, row in zip(unique[miss], sets[miss], rows):
                w = int(self.ages[s].argmin())
                self.tags[s, w] = key
                self.ages[s, w] = self.clock
                self.data[s * self.ways + w] = row
        self.hits += int(np.count_nonzero(hit[inverse]))
        if self.log_every and self.clock % self.log_every == 0:
            self.log()
        return out[inverse]

    def stats(self) -> dict:
        return {
            "lookups": self.clock,
            "accesses": self.accesses,
            "hits": self.hits,
            "hit_rate": self.hits / self.accesses if self.accesses else 0.0,
        }

    def log(self) -> None:
        if self.clock:
            logger.info("engram row cache: %s", json.dumps(self.stats()))


_SHARED: Optional[EngramRowCache] = None
_SHARED_LOCK = threading.Lock()


def shared_engram_row_cache(row_bytes: int) -> Optional[EngramRowCache]:
    """One cache for every Engram layer, sized by SGLANG_DSV41_ENGRAM_RAM_GIB."""
    global _SHARED
    budget = envs.SGLANG_DSV41_ENGRAM_RAM_GIB.get()
    if budget <= 0:
        return None
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = EngramRowCache.for_bytes(int(budget * _GIB), row_bytes)
            # Engine.shutdown() may kill the scheduler first; the periodic line covers that.
            atexit.register(_SHARED.log)
        elif _SHARED.row_bytes != row_bytes:
            raise ValueError(f"Engram layers disagree on row bytes: {_SHARED.row_bytes} vs {row_bytes}")
        return _SHARED
