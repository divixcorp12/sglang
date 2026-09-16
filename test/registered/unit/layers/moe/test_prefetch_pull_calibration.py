"""Contract tests for the graph-captured prefetch calibration observer."""

import os
import importlib.util
import json
import tempfile
from unittest import mock
from pathlib import Path

import torch
import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.serving.calibration import (
    PullCalibrationHistogram,
)


def _histogram() -> PullCalibrationHistogram:
    return PullCalibrationHistogram(layer_ids=(3,), device=torch.device("cpu"))


def test_histogram_uses_inclusive_unit_interval_boundaries():
    """A clamp or a last-bin off-by-one would corrupt score-band attribution."""
    histogram = _histogram()

    assert histogram.bin_index(torch.tensor(-0.1)) == 0
    assert histogram.bin_index(torch.tensor(0.0)) == 0
    assert histogram.bin_index(torch.tensor(1 / 256)) == 1
    assert histogram.bin_index(torch.tensor(255 / 256)) == 255
    assert histogram.bin_index(torch.tensor(1.0)) == 255
    assert histogram.bin_index(torch.tensor(1.1)) == 255


def test_target_outcomes_keep_route_value_separate_from_physical_demand_rows():
    """Counting covered routes as copied rows would make multi-route turns lie."""
    histogram = _histogram()
    histogram.stage(
        target_layer=3,
        candidate_id=torch.tensor([7]),
        top_score=torch.tensor(0.5),
        margin=torch.tensor(0.25),
        source_eligible=torch.tensor(True),
    )
    histogram.record_target(
        target_layer=3,
        flat_ids=torch.tensor([7, 7, 1]),
        missed_mask=torch.tensor([True, True, True]),
        physical_demand_rows=torch.tensor([2]),
    )

    record = histogram.snapshot()["layers"]["3"]
    assert record["score"]["opportunity"][128] == 1
    assert record["score"]["source_eligible"][128] == 1
    assert record["score"]["target_useful"][128] == 1
    assert record["score"]["target_wasted"][128] == 0
    assert record["score"]["physical_demand_rows"][128] == 2
    assert record["margin"]["target_useful"][64] == 1


def test_all_resident_and_no_candidate_do_not_create_a_false_useful_post():
    """A sentinel candidate must never become an apparent opportunity or hit."""
    histogram = _histogram()
    histogram.stage(
        target_layer=3,
        candidate_id=torch.tensor([-1]),
        top_score=torch.tensor(0.9),
        margin=torch.tensor(0.1),
        source_eligible=torch.tensor(False),
    )
    histogram.record_target(
        target_layer=3,
        flat_ids=torch.tensor([2, 3]),
        missed_mask=torch.tensor([False, False]),
        physical_demand_rows=torch.tensor([0]),
    )

    score = histogram.snapshot()["layers"]["3"]["score"]
    assert sum(score["opportunity"]) == 0
    assert sum(score["target_useful"]) == 0
    assert sum(score["target_wasted"]) == 0
    assert sum(score["physical_demand_rows"]) == 0


def test_reset_drops_graph_capture_warmup_counts_without_reallocating_state():
    """Capture replay must not be included in the profiling-run histogram."""
    histogram = _histogram()
    ids_address = histogram.candidate_ids.data_ptr()
    histogram.stage(3, torch.tensor([1]), torch.tensor(0.2), torch.tensor(0.1), torch.tensor(True))
    histogram.record_target(3, torch.tensor([1]), torch.tensor([True]), torch.tensor([1]))
    histogram.reset()

    score = histogram.snapshot()["layers"]["3"]["score"]
    assert histogram.candidate_ids.data_ptr() == ids_address
    assert sum(score["opportunity"]) == 0
    assert sum(score["physical_demand_rows"]) == 0


def test_calibration_collection_is_disabled_unless_explicitly_requested():
    """Timed arms must not accidentally pay for profiling-only observation."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION", None)
        assert envs.SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION.get() is False


def test_extended_summarizer_keeps_roles_fixed_and_bootstraps_paired_sessions():
    """Mixed run provenance or a shape-guessed metrics file would invalidate a comparison."""
    script = Path(__file__).parents[5] / "scripts/expert_prediction/prefetch/summarize_ab.py"
    spec = importlib.util.spec_from_file_location("summarize_ab", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = Path(tempfile.mkdtemp())

    def write(name, records):
        path = directory / name
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    b_results = write("b-results.jsonl", [
        {"session_id": "s1", "completion_tokens": 64, "decode_tokens_per_sec": 100, "ttft": 0.1},
        {"session_id": "s2", "completion_tokens": 64, "decode_tokens_per_sec": 50, "ttft": 0.2, "finish_reason": "length"},
    ])
    c_results = write("c-results.jsonl", [
        {"session_id": "s1", "completion_tokens": 64, "decode_tokens_per_sec": 50, "ttft": 0.1},
        {"session_id": "s2", "completion_tokens": 64, "decode_tokens_per_sec": 100 / 3, "ttft": 0.2},
    ])
    prediction = write("prediction.jsonl", [{"prefetch": {"observed_miss_rate": 0.25}}])
    hot = write("hot.jsonl", [{"counters": {"decode": {"0": {
        "resident_slots": 12, "side_pull_posted_rows": 4, "side_pull_useful_rows": 3,
        "side_pull_wasted_rows": 1,
    }}}}])
    manifest = write("manifest.json", [{"commit": "abc", "cache_size": 10240, "session_ids": ["s1", "s2"]}])
    b = module.summarize_arm(f"B={b_results}:{prediction}:{hot}:{manifest}")
    c = module.summarize_arm(f"C={c_results}:{prediction}:{hot}:{manifest}")

    assert b["median_decode_tok_s"] == 75
    assert b["p50_turn_decode_ms_per_token"] == 15
    assert b["p95_turn_decode_ms_per_token"] == 19.5
    assert b["truncation_count"] == 1
    assert b["observed_miss_rate"] == 0.25
    assert b["resident_slots"] == 12
    assert b["posted_rows"] == 4
    assert b["useful_precision"] == 0.75
    assert module.paired_session_bootstrap(b, c, seed=20260916, resamples=10_000) == [10.0, 10.0]
    with pytest.raises(ValueError, match="mixed commits"):
        c["manifest"]["commit"] = "def"
        module.paired_session_bootstrap(b, c, seed=20260916, resamples=1)
