"""b0_split.py: the B0 seeded session sample and its whole-session train/dev/calib/locked-test split. CPU only."""

import json
import random

SPLIT_FRACTIONS = (("train", 0.6), ("dev", 0.15), ("calib", 0.1), ("locked_test", 0.15))


def sample_sessions(path: str, *, n: int, seed: int) -> list[dict]:
    """``n`` rows of a sessions.jsonl drawn with ``seed``, in file order; refuses a file with fewer than ``n``."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) < n:
        raise ValueError(f"{path} holds {len(rows)} sessions, fewer than the {n} asked for")
    picked = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[i] for i in picked]


def split_sessions(ids: list[str], *, seed: int) -> dict[str, list[str]]:
    """Whole-session split, 180/45/30/45 at 300 ids: the same ids and seed give the same split, no id is in two."""
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate session ids")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    out: dict[str, list[str]] = {}
    start = 0
    for name, frac in SPLIT_FRACTIONS[:-1]:
        size = round(len(ids) * frac)
        out[name] = shuffled[start : start + size]
        start += size
    out[SPLIT_FRACTIONS[-1][0]] = shuffled[start:]
    return out
