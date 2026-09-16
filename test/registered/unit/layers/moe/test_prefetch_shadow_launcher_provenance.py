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
    assert manifest["trace"] is None
    assert "trace_report" not in manifest["paths"]


def test_trace_provenance_only_mode_records_diagnostic_trace_configuration(tmp_path):
    """Trace manifests describe the requested wrapper without running it."""
    run_dir = tmp_path / "trace-run"
    trace_command = "nsys profile --trace=cuda,nvtx,osrt"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "trace-provenance", "7999", "off"],
        cwd=ROOT,
        env={
            **os.environ,
            "RUN_KIND": "trace",
            "PREFETCH_TRACE_COMMAND": trace_command,
            "PREFETCH_PROVENANCE_ONLY": "1",
            "PREFETCH_WORKTREE": str(ROOT),
            "PREFETCH_RUN_DIR": str(run_dir),
            "MODEL_TOP_K": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    assert manifest["run_kind"] == "trace"
    assert manifest["trace"] == {
        "diagnostic_only": True,
        "command": trace_command,
        "report_path": str(run_dir / "trace" / "report"),
    }


def test_trace_mode_requires_explicit_wrapper_even_for_provenance_only(tmp_path):
    """A trace run cannot accidentally launch without its audited wrapper."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "missing-trace-command", "7999", "off"],
        cwd=ROOT,
        env={
            **os.environ,
            "RUN_KIND": "trace",
            "PREFETCH_PROVENANCE_ONLY": "1",
            "PREFETCH_WORKTREE": str(ROOT),
            "PREFETCH_RUN_DIR": str(tmp_path / "run"),
            "MODEL_TOP_K": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "PREFETCH_TRACE_COMMAND" in result.stderr


def test_trace_mode_rejects_whitespace_only_wrapper_even_for_provenance_only(tmp_path):
    """Tokenization must not turn whitespace into an empty executable."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "blank-trace-command", "7999", "off"],
        cwd=ROOT,
        env={
            **os.environ,
            "RUN_KIND": "trace",
            "PREFETCH_TRACE_COMMAND": "  \t  ",
            "PREFETCH_PROVENANCE_ONLY": "1",
            "PREFETCH_WORKTREE": str(ROOT),
            "PREFETCH_RUN_DIR": str(tmp_path / "run"),
            "MODEL_TOP_K": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "PREFETCH_TRACE_COMMAND" in result.stderr


def test_trace_mode_rejects_shell_quoted_wrapper_even_for_provenance_only(tmp_path):
    """The wrapper contract accepts simple argv tokens, not shell syntax."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "quoted-trace-command", "7999", "off"],
        cwd=ROOT,
        env={
            **os.environ,
            "RUN_KIND": "trace",
            "PREFETCH_TRACE_COMMAND": 'nsys profile --name="pcie trace"',
            "PREFETCH_PROVENANCE_ONLY": "1",
            "PREFETCH_WORKTREE": str(ROOT),
            "PREFETCH_RUN_DIR": str(tmp_path / "run"),
            "MODEL_TOP_K": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "simple whitespace-separated argv tokens" in result.stderr


def test_trace_mode_rejects_multiline_wrapper_before_tokenization(tmp_path):
    """A manifest must never describe wrapper text that the launcher ignores."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "multiline-trace-command", "7999", "off"],
        cwd=ROOT,
        env={
            **os.environ,
            "RUN_KIND": "trace",
            "PREFETCH_TRACE_COMMAND": "nsys profile\nignored-command",
            "PREFETCH_PROVENANCE_ONLY": "1",
            "PREFETCH_WORKTREE": str(ROOT),
            "PREFETCH_RUN_DIR": str(tmp_path / "run"),
            "MODEL_TOP_K": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "line breaks" in result.stderr


def test_non_trace_launch_contract_unsets_inherited_trace_report_path():
    """Timed and profiling launches cannot inherit a caller's trace destination."""
    launcher = LAUNCHER.read_text()

    assert 'launch_env=(env)' in launcher
    assert 'launch_env+=(-u PREFETCH_TRACE_REPORT_PATH)' in launcher
    assert 'exec flock --nonblock /data/models/slang/nvfp4-work/cc-gpu.lock "${launch_env[@]}"' in launcher


def test_trace_mode_refuses_calibration_even_for_provenance_only(tmp_path):
    """Trace diagnostics must not create calibration artifacts."""
    result = subprocess.run(
        ["bash", str(LAUNCHER), "trace-calibration", "7999", "off"],
        cwd=ROOT,
        env={
            **os.environ,
            "RUN_KIND": "trace",
            "PREFETCH_TRACE_COMMAND": "nsys profile",
            "PREFETCH_CALIBRATION": "1",
            "PREFETCH_PROVENANCE_ONLY": "1",
            "PREFETCH_WORKTREE": str(ROOT),
            "PREFETCH_RUN_DIR": str(tmp_path / "run"),
            "MODEL_TOP_K": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "trace" in result.stderr.lower()
    assert "calibration" in result.stderr.lower()
