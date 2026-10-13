"""L+2 cap-1 evaluation: per-point counts r1_metrics does not give (spec section 1.3, C6, C9-C12, D3).

Abstention and whole-layer coverage per operating point, the matched-candidate split of useful reads into rows the
gate's top-12 holds and rows outside it, and the DSpark breakdown (row sets, positions, live block width) pulled from
an r1_metrics.report_to_json dict. Shared experts count once, as r1_metrics counts them. No ordinary-decode dataset
exists, so none of this measures ordinary decode.

Inputs: R1Groups of one split and, per group, the expert ids a policy read (r1_policy.select output).
Output: plain dicts of ints and floats, ready for result.json.

Usage: l2_cap1_metrics.coverage_counts(groups, reads); l2_cap1_metrics.dspark_json(r1_metrics.report_to_json(rep))
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_rows  # noqa: E402
from r1_groups import NUM_EXPERTS, R1Group  # noqa: E402

_WIDTH_PREFIX = "width:"


def _check(groups: list[R1Group], reads: list[np.ndarray]) -> None:
    if len(groups) != len(reads):
        raise ValueError(f"{len(groups)} groups but {len(reads)} read sets")


def _never_mask(r: R1Group) -> np.ndarray:
    m = np.zeros(NUM_EXPERTS, dtype=bool)
    m[np.asarray(b1_rows.never_candidates(r.g), dtype=np.int64)] = True
    return m


def coverage_counts(groups: list[R1Group], reads: list[np.ndarray]) -> dict:
    """Abstention and whole-layer coverage of one policy's reads over (forward, target layer) groups."""
    _check(groups, reads)
    abstained = with_miss = fully = partly = 0
    for r, read in zip(groups, reads):
        abstained += int(len(read) == 0)
        n_elig = len(r.labels.eligible_all)
        if n_elig == 0:
            continue
        useful = int(r.labels.miss[np.asarray(read, dtype=np.int64)].sum())
        with_miss += 1
        fully += int(useful == n_elig)
        partly += int(0 < useful < n_elig)
    n = len(groups)
    return {"forward_layers": n, "abstained": abstained, "abstention_rate": abstained / n if n else 0.0,
            "layers_with_miss": with_miss, "fully_covered": fully, "partly_covered": partly,
            "not_covered": with_miss - fully - partly}


def candidate_split(groups: list[R1Group], reads: list[np.ndarray]) -> dict:
    """Useful reads and eligible rows split by whether the row is inside the gate's top-12 candidates."""
    _check(groups, reads)
    out = {"inside_top12": 0, "outside_top12": 0, "eligible_inside": 0, "eligible_outside": 0}
    for r, read in zip(groups, reads):
        never = _never_mask(r)
        miss = r.labels.miss
        useful = np.zeros(NUM_EXPERTS, dtype=bool)
        useful[np.asarray(read, dtype=np.int64)] = True
        useful &= miss
        out["outside_top12"] += int((useful & never).sum())
        out["inside_top12"] += int((useful & ~never).sum())
        out["eligible_outside"] += int((miss & never).sum())
        out["eligible_inside"] += int((miss & ~never).sum())
    return out


def gain_split(groups: list[R1Group], *, cand_reads: list[np.ndarray], gate_reads: list[np.ndarray]) -> dict:
    """D3: useful reads of a learned head and of the gate, inside and outside the gate's top-12, and the gain."""
    cand = candidate_split(groups, cand_reads)
    gate = candidate_split(groups, gate_reads)
    return {side: {"cand": cand[f"{side}_top12"], "gate": gate[f"{side}_top12"],
                   "diff": cand[f"{side}_top12"] - gate[f"{side}_top12"]} for side in ("inside", "outside")}


def dspark_json(report_json: dict) -> dict:
    """Row sets, coverage by draft position ('0' = anchor) and by live block width of one report_to_json dict."""
    rowsets = {k: {"eligible_all": v["eligible_all"], "useful": v["useful"], "recall": v["recall"]}
               for k, v in report_json["rowsets"].items()}
    positions = {k: {"miss_rows": v["miss_rows"], "covered": v["covered"], "coverage": v["coverage"]}
                 for k, v in report_json["positions"].items()}
    widths = {k[len(_WIDTH_PREFIX):]: {"useful": v["useful"], "eligible_all": v["eligible_all"], "recall": v["recall"]}
              for k, v in report_json["cells"].items() if k.startswith(_WIDTH_PREFIX)}
    return {"rowsets": rowsets, "positions": positions, "widths": widths}
