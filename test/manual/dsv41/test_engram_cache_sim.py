"""The Engram cache simulator's LRU math against a brute-force LRU."""

import importlib.util
import os
from collections import OrderedDict

import numpy as np
import pytest

pytest.importorskip("numba")

_SIM = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41", "engram_cache_sim.py"
)
_spec = importlib.util.spec_from_file_location("engram_cache_sim", _SIM)
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


def _brute_force_lru(keys, capacity):
    cache, hits = OrderedDict(), []
    for key in keys:
        if key in cache:
            cache.move_to_end(key)
            hits.append(True)
        else:
            hits.append(False)
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return np.array(hits)


def test_hand_computed_distances():
    distances, prev = sim.reuse_distances(np.array([0, 1, 0, 2, 1, 0], dtype=np.int64))
    assert distances.tolist() == [-1, -1, 1, -1, 2, 2]
    assert prev.tolist() == [-1, -1, 0, -1, 1, 2]


@pytest.mark.parametrize("capacity", [1, 2, 7, 50, 299, 300, 1000])
def test_matches_brute_force_lru(capacity):
    keys = np.random.default_rng(0).zipf(1.3, size=5000).astype(np.int64) % 300
    distances, _ = sim.reuse_distances(keys)
    expected = _brute_force_lru(keys.tolist(), capacity)
    assert np.array_equal(sim.lru_hits(distances, capacity), expected)


def test_session_repeat_hits():
    keys = np.array([4, 4, 9, 4, 9], dtype=np.int64)
    _, prev = sim.reuse_distances(keys)
    session_start = np.array([0, 0, 0, 3, 3])
    assert sim.session_repeat_hits(prev, session_start).tolist() == [
        False, True, False, False, False,
    ]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
