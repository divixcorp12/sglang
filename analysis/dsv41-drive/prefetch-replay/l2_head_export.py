"""Export a learned L+2 checkpoint as the one folded-head file the serving hook loads (SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD).

Folds the checkpoint (l2_fast_torch.fold) over the serving rows of the dev and calib groups, stores it in the kernels'
precision (bf16 matrices, fp32 biases and tables), and writes one safetensors file with the checkpoint sha256, the F109
frozen calib threshold and where it came from. The threshold is read, never refit (D2).

Inputs: a finished R1 run (exclusions, stages/l2/result.json), the B1 dataset and the F109 output directory (result.json,
read only). Output: --out only; it may not be RUN_DIR, inside it, or inside any protected run directory.

Usage: l2_head_export.py RUN_DIR --dataset D --f109 F109_DIR --out PATH [--scorer L2_T3] [--budget 0.75]
Exit: 0 done, 1 failure, 2 refused --out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import l2_fast_common as common  # noqa: E402
import l2_fast_kernels  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_models  # noqa: E402

from sglang.srt.layers.moe import ram_prefetch_learned_head as head_file  # noqa: E402

SCORER = "L2_T3"
BUDGET = "0.75"
SPLITS = ("dev", "calib")


def export_head(model: r1_models.R1Model, *, rows: tuple[l2_fast_torch.FoldRow, ...], checkpoint_sha256: str,
                threshold: float, threshold_source: dict, out: str) -> head_file.HeadMeta:
    """Fold ``model`` over ``rows`` and write the head file to ``out``; returns the metadata written."""
    if threshold is None or not math.isfinite(threshold):
        raise ValueError(f"threshold must be a finite float, got {threshold!r}")
    strata = model.head.vocab.strata
    folded = l2_fast_torch.fold(model, rows=rows)
    weights = l2_fast_kernels.to_kernel_weights(folded, device="cpu")
    meta = head_file.HeadMeta(
        format_version=head_file.HEAD_FORMAT_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        threshold=float(threshold),
        threshold_source=threshold_source,
        horizon=head_file.HEAD_HORIZON,
        rows=tuple(head_file.HeadRow(source_layer=r.source_layer, target_layer=r.target_layer,
                                     stratum=strata.index(r.stratum)) for r in rows),
    )
    tensors = {name: getattr(weights, name) for name in head_file.TENSOR_NAMES}
    head_file.write_head_file(out, tensors=tensors, meta=meta)
    return meta


def _frozen_threshold(f109_dir: str, *, scorer: str, budget: str) -> tuple[float, dict]:
    import l2_fast_parity  # heavy import chain (expert-stream kernels); only the CLI needs it

    path = os.path.join(f109_dir, "result.json")
    with open(path) as f:
        value = l2_fast_parity.frozen_thresholds(json.load(f))[scorer][budget]
    if value is None:
        raise ValueError(f"{path}: no feasible calib threshold for {scorer} at budget {budget}")
    return float(value), {"path": path, "split": "calib", "scorer": scorer, "budget": budget}


def _serving_rows(*, run_dir: str, dataset: str) -> tuple[l2_fast_torch.FoldRow, ...]:
    import l2_fast_parity
    import r1_arms
    import r1_groups

    exclusions = r1_groups.load_exclusions(os.path.join(run_dir, r1_arms.EXCLUSIONS_REL))
    groups = [r for split in SPLITS
              for r in l2_fast_parity._read_split(dataset, split=split, exclusions=exclusions, limit=None)]
    return l2_fast_torch.serving_rows(groups)


def run(*, run_dir: str, dataset: str, f109_dir: str, out: str, scorer: str, budget: str) -> dict:
    import l2_cap1_scorers

    with open(os.path.join(run_dir, "stages", "l2", "result.json")) as f:
        ckpt_path, recorded_sha = l2_cap1_scorers.learned_ckpts(json.load(f))[scorer]
    sha = l2_cap1_scorers.verify_ckpt(ckpt_path, sha=recorded_sha)
    threshold, source = _frozen_threshold(f109_dir, scorer=scorer, budget=budget)
    rows = _serving_rows(run_dir=run_dir, dataset=dataset)
    model = r1_models.load_r1(ckpt_path)[0].cpu()
    meta = export_head(model, rows=rows, checkpoint_sha256=sha, threshold=threshold, threshold_source=source, out=out)
    return {"out": out, "checkpoint": ckpt_path, "target_layers": [r.target_layer for r in meta.rows],
            "threshold": meta.threshold, "checkpoint_sha256": sha}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("--dataset", required=True)
    p.add_argument("--f109", required=True, help="F109 output directory; only its result.json is read")
    p.add_argument("--out", required=True, help="the head file to write")
    p.add_argument("--scorer", default=SCORER, choices=("L2_T0", "L2_T3"))
    p.add_argument("--budget", default=BUDGET)
    return p


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    try:
        out = common.guard_out(a.out, run_dir=a.run_dir)
    except common.RefusedOut as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    result = run(run_dir=a.run_dir, dataset=a.dataset, f109_dir=a.f109, out=out, scorer=a.scorer, budget=a.budget)
    print(msgspec.json.encode(result).decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
