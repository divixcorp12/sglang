"""Which DSpark draft experts stay on the GPU: a calibration file made from a routes probe.

The probe (``SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH``) logs each draft MoE call's topk ids per stage (``layer_id``);
``top_n_resident_set`` keeps each stage's N most-routed experts. Every other routed expert of the stage runs on the
CPU (``cpu_experts/draft.py``).
"""

import collections
import json
import threading
from typing import Iterable, Mapping, Sequence

from sglang.srt.environ import envs

VERSION = 1

_cache: dict[str, dict[int, frozenset[int]]] = {}
_lock = threading.Lock()


def top_n_resident_set(route_lines: Iterable[str], n: int) -> dict[int, list[int]]:
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in route_lines:
        record = json.loads(line)
        counts[int(record["layer"])].update(e for row in record["ids"] for e in row if e >= 0)
    return {
        layer: [e for e, _ in sorted(c.items(), key=lambda item: (-item[1], item[0]))[:n]]
        for layer, c in sorted(counts.items())
    }


def write_resident_set(path: str, stages: Mapping[int, Sequence[int]], *, n: int, source: str) -> None:
    body = {
        "version": VERSION,
        "n": n,
        "source": source,
        "stages": {str(layer): [int(e) for e in ids] for layer, ids in sorted(stages.items())},
    }
    with open(path, "w") as f:
        json.dump(body, f, indent=1)
        f.write("\n")


def load_resident_set(path: str) -> dict[int, frozenset[int]]:
    try:
        with open(path) as f:
            body = json.load(f)
    except (OSError, ValueError) as error:
        raise ValueError(f"DSpark draft resident set {path}: {error}") from error
    if not isinstance(body, dict) or body.get("version") != VERSION or not isinstance(body.get("stages"), dict):
        raise ValueError(f"DSpark draft resident set {path}: expected version {VERSION} with a 'stages' map")
    stages = {}
    for key, ids in body["stages"].items():
        if not key.isdigit() or not isinstance(ids, list):
            raise ValueError(f"DSpark draft resident set {path}: stage {key!r} is not a layer id with a list")
        if any(not isinstance(e, int) or e < 0 for e in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"DSpark draft resident set {path}: stage {key} ids must be distinct ints >= 0")
        stages[int(key)] = frozenset(ids)
    return stages


def resident_for(layer_id: int) -> frozenset[int]:
    path = envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.get()
    if not path:
        return frozenset()
    with _lock:
        if path not in _cache:
            _cache[path] = load_resident_set(path)
        stages = _cache[path]
    if layer_id not in stages:
        raise ValueError(
            f"DSpark draft resident set {path} has no stage {layer_id} (it lists {sorted(stages)}); "
            "recalibrate it from this draft's routes"
        )
    return stages[layer_id]
