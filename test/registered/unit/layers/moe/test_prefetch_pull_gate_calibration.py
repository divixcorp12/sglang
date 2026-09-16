"""Offline selection tests for a provenance-locked width-one pull gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[5] / "scripts/expert_prediction/prefetch/calibrate_pull_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("calibrate_pull_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _histogram(*, session: str, useful_score_bin: int, useful_margin_bin: int = 0) -> dict:
    """One BS1/top-k-unique profiling artifact with a hand-checkable trace."""
    bins = 256

    def feature(useful_bin: int) -> dict[str, list[int]]:
        opportunity = [0] * bins
        useful = [0] * bins
        wasted = [0] * bins
        demand = [0] * bins
        source = [0] * bins
        # At high score, 4 useful pulls; at low score, 4 wasted pulls.  Both
        # carry physical target demand rows so route counts cannot substitute.
        opportunity[useful_bin] = useful[useful_bin] = source[useful_bin] = 4
        demand[useful_bin] = 4
        low = 32
        opportunity[low] = wasted[low] = source[low] = 4
        demand[low] = 12
        return {
            "opportunity": opportunity,
            "source_eligible": source,
            "target_useful": useful,
            "target_wasted": wasted,
            "physical_demand_rows": demand,
        }

    return {
        "schema_version": 1,
        "bin_count": bins,
        "range": [0.0, 1.0],
        "complete": True,
        "provenance": {
            "commit": "abc123",
            "predictor": "llapor",
            "checkpoint_checksum": "weights-sha",
            "cache_size": 10240,
            "session_ids": [session],
            "session_set_checksum": f"sessions-{session}",
            "batch_size": 1,
            "top_k_unique": True,
            "shape_provenance": "BS1",
            "top_k": "top-k-unique",
        },
        "layers": {"3": {"score": feature(useful_score_bin), "margin": feature(useful_margin_bin)}},
    }


def _costs() -> dict:
    return {
        "schema_version": 1,
        "layers": {
            "3": {
                "useful_row_value": 10.0,
                "posted_row_cost": 1.0,
                "physical_demand_row_cost": 0.5,
                "scorer_plus_selection_cost": 0.0,
                "count_zero_control_cost": 0.0,
                "join_exposure_cost": 0.0,
            }
        },
    }


def test_selects_positive_heldout_top_score_threshold_and_emits_complete_artifact():
    """A selector that ignores physical rows or reuses train sessions is unsafe."""
    module = _module()
    artifact = module.calibrate_gate(
        _histogram(session="train", useful_score_bin=224),
        _histogram(session="heldout", useful_score_bin=224),
        _costs(),
    )

    layer = artifact["layers"]["3"]
    assert artifact["complete"] is True
    assert artifact["predictor"] == "llapor"
    assert artifact["target_layers"] == [3]
    assert artifact["training_session_set_checksum"] == "sessions-train"
    assert artifact["heldout_session_set_checksum"] == "sessions-heldout"
    assert layer["feature"] == "top_score"
    assert layer["threshold"] == pytest.approx(224 / 256)
    # 4 useful rows * 10 - 4 posted rows * 1 - 4 physical demand rows * .5.
    # An implementation that substitutes route count or drops demand cost gets 36.
    assert layer["training_net_value"] == pytest.approx(34.0)
    assert layer["heldout_net_value"] == pytest.approx(34.0)
    assert layer["training_physical_rows"] == 8


def test_rejects_mixed_provenance_and_shared_session_sets():
    module = _module()
    train = _histogram(session="train", useful_score_bin=224)
    heldout = _histogram(session="heldout", useful_score_bin=224)
    heldout["provenance"]["cache_size"] = 5120
    with pytest.raises(ValueError, match="cache_size"):
        module.calibrate_gate(train, heldout, _costs())

    heldout = _histogram(session="train", useful_score_bin=224)
    with pytest.raises(ValueError, match="session"):
        module.calibrate_gate(train, heldout, _costs())


def test_rejects_incomplete_or_out_of_scope_histograms_before_selection():
    module = _module()
    incomplete = _histogram(session="train", useful_score_bin=224)
    incomplete["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        module.calibrate_gate(incomplete, _histogram(session="heldout", useful_score_bin=224), _costs())

    wrong_scope = _histogram(session="train", useful_score_bin=224)
    wrong_scope["provenance"]["batch_size"] = 2
    with pytest.raises(ValueError, match="BS1"):
        module.calibrate_gate(wrong_scope, _histogram(session="heldout", useful_score_bin=224), _costs())


def test_refuses_to_emit_gate_when_no_positive_train_threshold_survives_heldout():
    module = _module()
    heldout = _histogram(session="heldout", useful_score_bin=64)
    for feature in ("score", "margin"):
        counters = heldout["layers"]["3"][feature]
        counters["target_wasted"] = list(counters["opportunity"])
        counters["target_useful"] = [0] * 256
    with pytest.raises(ValueError, match="held-out"):
        module.calibrate_gate(
            _histogram(session="train", useful_score_bin=224),
            heldout,
            _costs(),
        )


def test_cli_writes_only_a_complete_provenance_locked_gate(tmp_path):
    module = _module()
    train, heldout, costs, out = (tmp_path / name for name in ("train.json", "heldout.json", "costs.json", "gate.json"))
    train.write_text(json.dumps(_histogram(session="train", useful_score_bin=224)))
    heldout.write_text(json.dumps(_histogram(session="heldout", useful_score_bin=224)))
    costs.write_text(json.dumps(_costs()))
    module.main(["--train", str(train), "--heldout", str(heldout), "--costs", str(costs), "--out", str(out)])
    assert json.loads(out.read_text())["complete"] is True
