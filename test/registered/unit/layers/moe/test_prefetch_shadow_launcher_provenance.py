"""No-server contract tests for the shadow launcher's calibration provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[5]
LAUNCHER = ROOT / "scripts/expert_prediction/run-shadow-server.sh"


def test_provenance_only_mode_emits_canonical_bs1_topk_unique_metadata(tmp_path):
    """Offline gates must not receive shell-rendered scope strings."""
    run_dir = tmp_path / "run"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "provenance", "7999", "off"],
        cwd=ROOT,
        env={
            **os.environ,
            "RUN_KIND": "profiling",
            "PREFETCH_CALIBRATION": "1",
            "PREFETCH_PROVENANCE_ONLY": "1",
            "PREFETCH_WORKTREE": str(ROOT),
            "PREFETCH_RUN_DIR": str(run_dir),
            "SESSION_IDS_JSON": '["training-session"]',
            "SESSION_SET_CHECKSUM": "training-sha",
            "PREFETCH_CHECKPOINT_CHECKSUM": "checkpoint-sha",
            "MODEL_TOP_K": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    expected_shape = {
        "batch_size": 1,
        "top_k": 10,
        "top_k_unique": True,
        "cuda_graph_decode_batch_size": 1,
        "cuda_graph_max_decode_batch_size": 1,
    }
    assert manifest["batch_size"] == 1
    assert manifest["top_k"] == 10
    assert manifest["top_k_unique"] is True
    assert manifest["shape_provenance"] == expected_shape
    calibration = manifest["calibration_provenance"]
    assert calibration["batch_size"] == 1
    assert calibration["top_k"] == 10
    assert calibration["top_k_unique"] is True
    assert calibration["shape_provenance"] == expected_shape
