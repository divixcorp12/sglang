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
import time
from typing import Callable, Optional

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# One lookup per Engram layer per forward, so this logs about every 256 forwards.
LOG_EVERY_LOOKUPS = 512
# Cache-counter snapshots are written at most this often per source.
SNAPSHOT_INTERVAL_S = 0.5


class CacheStatsSink:
    """Appends time-stamped cumulative cache counters to a JSONL side file.

    The line-buffered file sits beside the expert trace (``<trace>.cache-stats``)
    and carries the same ``time.monotonic()`` clock, so a run can cut the counters
    at session boundaries. A source calls ``maybe_write`` from its hot path; the
    snapshot is built only when the source's interval has passed, so a source costs
    one clock read per call while the sink exists and nothing when it does not.
    """

    def __init__(self, path: str, interval_s: float = SNAPSHOT_INTERVAL_S) -> None:
        self._file = open(path, "a", buffering=1)
        self._interval_s = interval_s
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def maybe_write(self, kind: str, snapshot: Callable[[], dict], force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next.get(kind, 0.0):
            return
        with self._lock:
            self._next[kind] = now + self._interval_s
            line = {"kind": kind, "t": round(now, 6), **snapshot()}
            self._file.write(json.dumps(line) + "\n")


_SINK: Optional[CacheStatsSink] = None
_SINK_PATH = ""


def cache_stats_sink() -> Optional[CacheStatsSink]:
    """The process-wide sink when SGLANG_DSV41_EXPERT_TRACE_PATH is set, else None."""
    global _SINK, _SINK_PATH
    trace_path = envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get()
    if not trace_path:
        return None
    path = trace_path + ".cache-stats"
    if _SINK is None or _SINK_PATH != path:
        _SINK, _SINK_PATH = CacheStatsSink(path), path
    return _SINK


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
        # Distinct rows fetched from the backing table, ways that held a row
        # another key replaced, and ways ever filled: the row-level counters
        # ``accesses``/``hits`` (which count a repeated key each time) cannot give.
        self.misses = 0
        self.evictions = 0
        self.filled_rows = 0
        self._sink = cache_stats_sink()

    @classmethod
    def for_bytes(cls, budget_bytes: int, row_bytes: int) -> EngramRowCache:
        return cls(budget_bytes // row_bytes, row_bytes)

    def lookup(self, keys: np.ndarray, fetch: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        rows, inverse = self._lookup_unique(keys, fetch)
        return rows[inverse]

    def lookup_into(
        self,
        keys: np.ndarray,
        fetch: Callable[[np.ndarray], np.ndarray],
        destination: np.ndarray,
    ) -> None:
        """Fill caller-owned packed row storage in request order."""
        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        destination = np.asarray(destination)
        if destination.shape != (keys.size, self.row_bytes) or destination.dtype != np.uint8:
            raise ValueError(
                f"destination must be uint8 [{keys.size}, {self.row_bytes}], got "
                f"{destination.dtype} {destination.shape}"
            )
        rows, inverse = self._lookup_unique(keys, fetch)
        for i, unique_index in enumerate(inverse):
            destination[i] = rows[unique_index]

    def _lookup_unique(
        self, keys: np.ndarray, fetch: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
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
                if self.tags[s, w] < 0:
                    self.filled_rows += 1
                else:
                    self.evictions += 1
                self.tags[s, w] = key
                self.ages[s, w] = self.clock
                self.data[s * self.ways + w] = row
        self.hits += int(np.count_nonzero(hit[inverse]))
        self.misses += int(np.count_nonzero(miss))
        if self._sink is not None:
            self._sink.maybe_write("engram", self.stats)
        if self.log_every and self.clock % self.log_every == 0:
            self.log()
        return out, inverse

    def stats(self) -> dict:
        return {
            "lookups": self.clock,
            "accesses": self.accesses,
            "hits": self.hits,
            "hit_rate": self.hits / self.accesses if self.accesses else 0.0,
            "misses": self.misses,
            "evictions": self.evictions,
            "filled_rows": self.filled_rows,
            "capacity_rows": self.n_sets * self.ways,
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
