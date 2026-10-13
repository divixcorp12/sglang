"""C2 gate: cap-2 reproduction check of the saved spike-l2 points.

Inputs: evaluate_split results at cap 2 and mean budget 1.0 for the spike's scorers, plus the loaded
spike-l2/result.json. Output: a status dict, or SanityError on any mismatch. Writes nothing.
Usage: import l2_cap1_sanity; l2_cap1_sanity.check_spike(evals, spike=spike).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_l2_spike  # noqa: E402

SANITY_SCORERS = ("gate", "L2_T0", "L2_T3")
FIELDS = ("reads", "useful", "eligible")
SPLITS = ("dev", "calib")
LABEL = "observed offline"


class SanityError(RuntimeError):
    pass


def spike_scorers(*, l2: dict, f0, dataset: str, split: str, device: str) -> list:
    return r1_l2_spike._scorers(l2=l2, f0=f0, dataset=dataset, split=split, device=device)


def _lookup(tree: dict, *keys):
    for k in keys:
        if not isinstance(tree, dict) or k not in tree:
            return None
        tree = tree[k]
    return tree


def _mismatches(*, evals: dict, spike: dict, cap: int) -> tuple[list[str], list[str]]:
    bad: list[str] = []
    checked: list[str] = []
    for split in SPLITS:
        for name in SANITY_SCORERS:
            tag = f"{split}/{name}"
            got = _lookup(evals, split, name, cap, "point")
            want = _lookup(spike, "points", split, name)
            if got is None or want is None:
                bad.append(f"{tag}: missing ({'run' if got is None else 'spike'})")
                continue
            checked.append(tag)
            for f in FIELDS:
                if f not in got or f not in want:
                    bad.append(f"{tag}.{f}: missing")
                elif got[f] != want[f]:
                    bad.append(f"{tag}.{f}: run={got[f]!r} spike={want[f]!r}")
    return bad, checked


def check_spike(evals: dict, *, spike: dict, cap: int = 2) -> dict:
    bad, checked = _mismatches(evals=evals, spike=spike, cap=cap)
    if bad:
        raise SanityError("cap-2 sanity mismatch vs spike-l2: " + "; ".join(bad))
    return {"checked": checked, "fields": list(FIELDS), "label": LABEL, "status": "reproduced"}
