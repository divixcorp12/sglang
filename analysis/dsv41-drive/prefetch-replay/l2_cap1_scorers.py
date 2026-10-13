"""L+2 cap-1 scorers: the D2 native gate (best margin over tokens of each token's best available expert), the C7
serving-coverage filter and the D1 learned scorers (T0, T3, T0 best_cap1) loaded with sha256 verification.

Inputs: R1Groups, the saved stages/l2/result.json dict, the F0 head, dataset, split and device.
Output: GroupScorers for r1_policy.score_split; (covered, uncovered) group lists.

Usage: scorers = l2_cap1_scorers.cap1_scorers(l2_result=l2, f0=f0, dataset=ds, split="dev", device="cuda")
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
import b1_baselines  # noqa: E402
import b1_head  # noqa: E402
import b1_score  # noqa: E402
import r1_arms  # noqa: E402
import r1_cache  # noqa: E402
import r1_groups  # noqa: E402
import r1_models  # noqa: E402
import r1_stage  # noqa: E402
from r1_groups import GroupScorer, R1Group  # noqa: E402

GATE = "gate_cap1"
LEARNED = ("L2_T0", "L2_T3", "L2_T0_best_cap1")
_BEST_ARM = "T0"
_BEST_KEY = "best_cap1"


class GateTop1Scorer:
    """Native gate at cap 1: each token offers only its best available expert; the block keeps the best margin."""

    def __init__(self) -> None:
        self._scorer = b1_baselines.GateScorer(floors=None)
        self.name = GATE

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        out = []
        for r in groups:
            scores = np.asarray(self._scorer(r.g).scores, dtype=np.float32)
            block = b1_score._margin_values(scores, skip=r1_groups.skip_mask(r.g), k=1)
            out.append(block.astype(np.float32))
        return out


def split_covered(groups: list[R1Group]) -> tuple[list[R1Group], list[R1Group]]:
    """(covered, uncovered) by the target row's biased gate, order preserved."""
    covered = [r for r in groups if r.g.gate_has_bias]
    uncovered = [r for r in groups if not r.g.gate_has_bias]
    return covered, uncovered


def learned_ckpts(l2_result: dict) -> dict[str, tuple[str, str]]:
    """Scorer name -> (checkpoint path, recorded sha256) for the three learned rows."""
    arms = l2_result["arms"]
    best = arms[_BEST_ARM]["extra_ckpts"][_BEST_KEY]
    return {
        "L2_T0": (arms["T0"]["ckpt_path"], arms["T0"]["ckpt_sha"]),
        "L2_T3": (arms["T3"]["ckpt_path"], arms["T3"]["ckpt_sha"]),
        "L2_T0_best_cap1": (best["path"], best["sha"]),
    }


def verify_ckpt(path: str, *, sha: str) -> str:
    actual = r1_stage.sha256_file(path)
    if actual != sha:
        raise ValueError(f"{path}: sha256 {actual} differs from the recorded {sha}")
    return actual


def cap1_scorers(*, l2_result: dict, f0: b1_head.B1Head, dataset: str, split: str, device: str) -> list[GroupScorer]:
    ckpts = learned_ckpts(l2_result)
    for path, sha in ckpts.values():
        verify_ckpt(path, sha=sha)
    x_of = r1_cache.make_x_of(dataset, split=split)

    def collate(gs):
        return r1_cache.collate(gs, x_of=x_of, pca=f0.pca, cfg=f0.cfg, vocab=f0.vocab, device=device)

    out: list[GroupScorer] = [GateTop1Scorer()]
    for name in LEARNED:
        model = r1_models.load_r1(ckpts[name][0])[0].to(device)
        scorer = r1_models.R1GroupScorer(model, collate_fn=collate, name=name)
        out.append(r1_arms._Chunked(scorer, chunk=r1_arms._CALIB_CHUNK))
    return out
