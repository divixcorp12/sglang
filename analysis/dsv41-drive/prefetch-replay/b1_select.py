"""B1 selection and verdict: the dev grid, the calib cuts, the frozen selection file and the L+1 bar (S1.2, R1).

Model and threshold selection read dev and calib tables only. freeze_selection writes the chosen config, cuts and model
checksums once; require_frozen refuses the locked_test split until that file exists, and the final run records its
sha256. The session bootstrap, the learning-curve 'still improving' rule (D5) and the verdict arms live here too.

Inputs: b1_score.Table objects (dev for the grid, a calib cut sweep for the cuts) and per-session paired differences.
Usage: import b1_select; sha = b1_select.require_frozen("locked_test", "/mnt/nvme1/nvfp4-work/b1-run-X/selection.json")
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import sys

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
from b1_score import READS, Table  # noqa: E402

LOCKED_TEST = "locked_test"
DEV, CALIB = "dev", "calib"
PCA_RANKS = (128, 256)
HIDDENS = (256, 512)
HARD_NEG_WEIGHTS = (1.0, 2.0, 4.0)
# The B2 bar, Design section 2 / S1.2: +0.15 precision or 1.25x recall at 1 read, CI above 0, 0.1 ms per layer.
PRECISION_GAIN = 0.15
RECALL_RATIO = 1.25
COST_MS_PER_LAYER = 0.1
_EPS = 1e-9  # thresholds are inclusive; absorbs float error in differences such as 0.65 - 0.5
CEILING = "the native gate is the ceiling for same-token L+1 prediction at this budget"


class HorizonSelection(msgspec.Struct, frozen=True):
    variant: str
    config: dict
    cuts: dict[str, float]
    model_path: str
    model_sha256: str


class Selection(msgspec.Struct, frozen=True):
    horizons: dict[int, HorizonSelection]
    pop_path: str
    pop_sha256: str
    created: str


def grid(*, variant: str) -> list[dict]:
    """HeadConfig kwargs for the dev grid: PCA rank x hidden, and for E5 the hard-negative weight."""
    if variant == "E2":
        weights: tuple[float, ...] = (1.0,)
    elif variant == "E5":
        weights = HARD_NEG_WEIGHTS
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return [{"variant": variant, "pca_rank": r, "hidden": h, "hard_neg_weight": w}
            for r, h, w in itertools.product(PCA_RANKS, HIDDENS, weights)]


def _pooled(table: Table, budget: str):
    return table.cells[budget]["pooled"]


def pick_best(dev_tables: dict[str, Table]) -> str:
    """The key whose table has the highest pooled precision at reads=1 on dev; ties go to the smaller key."""
    if not dev_tables:
        raise ValueError("no dev tables to pick from")
    for key, t in dev_tables.items():
        if t.split != DEV:
            raise ValueError(f"selection reads dev only; table {key!r} is split {t.split!r}")
    return min(dev_tables, key=lambda key: (-_pooled(dev_tables[key], "reads=1").precision, key))


def calibrate_cuts(table_sweep: Table, *, reads=READS) -> dict[str, float]:
    """Per 'reads=r', the swept cut whose pooled mean reads per (forward, layer) is closest to r; ties to the higher cut."""
    if table_sweep.split != CALIB:
        raise ValueError(f"cuts are calibrated on calib only; table is split {table_sweep.split!r}")
    sweep = []
    for budget, keyed in table_sweep.cells.items():
        if budget.startswith("cut=") and "pooled" in keyed and keyed["pooled"].forwards:
            cell = keyed["pooled"]
            sweep.append((float(budget[len("cut="):]), cell.read / cell.forwards))
    if not sweep:
        raise ValueError("table has no cut sweep")
    return {f"reads={r}": min(sweep, key=lambda cm: (abs(cm[1] - r), -cm[0]))[0] for r in reads}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_selection(path: str, sel: Selection) -> str:
    """Write the selection once (JSON, indent=1) and return its sha256; an existing file is never overwritten."""
    data = msgspec.json.format(msgspec.json.encode(sel), indent=1) + b"\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)
    return _sha256_bytes(data)


def load_frozen(path: str) -> tuple[Selection, str]:
    with open(path, "rb") as f:
        data = f.read()
    return msgspec.json.decode(data, type=Selection), _sha256_bytes(data)


def require_frozen(split: str, frozen_path: str | None) -> str | None:
    """The frozen file's sha256 when it exists; locked_test without a valid frozen file raises PermissionError (R1)."""
    if frozen_path is None or not os.path.isfile(frozen_path):
        if split == LOCKED_TEST:
            raise PermissionError(f"locked_test is read only after selection is frozen; no file at {frozen_path!r}")
        return None
    try:
        return load_frozen(frozen_path)[1]
    except msgspec.DecodeError as e:
        if split == LOCKED_TEST:
            raise PermissionError(f"frozen selection {frozen_path!r} does not decode: {e}") from e
        raise


def paired_diffs(head: Table, gate: Table, *, budget: str = "reads=1") -> dict[str, float]:
    """Per-session head minus gate precision at ``budget``, over sessions both tables hold."""
    h, g = head.cells[budget], gate.cells[budget]
    return {key[len("session:"):]: h[key].precision - g[key].precision
            for key in sorted(h) if key.startswith("session:") and key in g}


def session_bootstrap(diffs_by_session: dict[str, float], *, n: int = 10_000,
                      seed: int = 0) -> tuple[float, float, float]:
    """(median, lo95, hi95): the paired median and a percentile 95% interval over sessions resampled with replacement."""
    if not diffs_by_session:
        raise ValueError("no sessions to bootstrap")
    diffs = np.array([diffs_by_session[s] for s in sorted(diffs_by_session)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(diffs[rng.integers(0, len(diffs), size=(n, len(diffs)))], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(np.median(diffs)), float(lo), float(hi)


def still_improving(p15: float, p30: float, p60: float) -> bool:
    """D5: the 30->60 session step raises dev precision by more than half the 15->30 step."""
    return (p60 - p30) > 0.5 * (p30 - p15)


def _recall_arm(gate: float | None, head: float | None) -> bool:
    if gate is None or head is None or not math.isfinite(gate) or not math.isfinite(head):
        return False
    return head > 0 and head >= RECALL_RATIO * gate - _EPS


def verdict(*, gate_floor_precision_r1: float, head_precision_r1: float, gate_floor_recall_at_p: float | None,
            head_recall_at_p: float | None, ci_lo: float, cost_ms_per_layer: float) -> dict:
    """The L+1 advance / do-not-advance call against the B2 bar."""
    arms = {
        "precision": head_precision_r1 - gate_floor_precision_r1 >= PRECISION_GAIN - _EPS,
        "recall": _recall_arm(gate_floor_recall_at_p, head_recall_at_p),
        "ci": ci_lo > 0,
        "cost": cost_ms_per_layer <= COST_MS_PER_LAYER + _EPS,
    }
    advance = (arms["precision"] or arms["recall"]) and arms["ci"] and arms["cost"]
    reasons = [
        f"precision at 1 read {head_precision_r1:.4f} vs gate+floors {gate_floor_precision_r1:.4f} "
        f"(needs +{PRECISION_GAIN}): {'met' if arms['precision'] else 'not met'}",
        f"recall at gate+floors precision {head_recall_at_p} vs {gate_floor_recall_at_p} "
        f"(needs {RECALL_RATIO}x): {'met' if arms['recall'] else 'not met'}",
        f"bootstrap 95% lower bound {ci_lo:.4f} (needs > 0): {'met' if arms['ci'] else 'not met'}",
        f"scorer cost {cost_ms_per_layer:.4f} ms per layer (needs <= {COST_MS_PER_LAYER}): "
        f"{'met' if arms['cost'] else 'not met'}",
    ]
    if not advance:
        reasons.append(f"do not advance: {CEILING}")
    return {"advance": advance, "arms": arms, "reasons": reasons}
