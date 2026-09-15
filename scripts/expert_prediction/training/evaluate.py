#!/usr/bin/env python
"""Summarize LLaPor/APEX per-layer manifests from a training run into a
condensed report (used for the offline-training write-up)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_manifests(run_dir: Path, model: str) -> list[dict]:
    manifests = []
    for manifest_path in sorted((run_dir / model).glob("*/manifest.json")):
        manifests.append(json.loads(manifest_path.read_text()))
    return manifests


def summarize_llapor(run_dir: Path) -> dict:
    rows = []
    for manifest in _load_manifests(run_dir, "llapor"):
        source_layer = manifest["grouping"]["source_layer"]
        for split_phase, values in manifest["metrics"].items():
            if split_phase not in ("dev_decode", "dev_prefill", "shifted_test_decode", "shifted_test_prefill"):
                continue
            rows.append({"source_layer": source_layer, "split_phase": split_phase, **values})
    return {"pairs": len(_load_manifests(run_dir, "llapor")), "rows": rows}


def summarize_apex(run_dir: Path) -> dict:
    rows = []
    for manifest in _load_manifests(run_dir, "apex"):
        layer_id = manifest["layer_id"]
        rows.append(
            {
                "layer_id": layer_id,
                "dev_kl": manifest["metrics"]["dev_kl"],
                "calibration": manifest["metrics"]["calibration"],
                "coverage": manifest["metrics"]["coverage"],
            }
        )
    dev_kls = [row["dev_kl"] for row in rows]
    return {
        "layers": len(rows),
        "mean_dev_kl": statistics.fmean(dev_kls) if dev_kls else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", choices=("llapor", "apex", "both"), default="both")
    args = parser.parse_args()
    report = {}
    if args.model in ("llapor", "both"):
        report["llapor"] = summarize_llapor(args.run_dir)
    if args.model in ("apex", "both"):
        report["apex"] = summarize_apex(args.run_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
