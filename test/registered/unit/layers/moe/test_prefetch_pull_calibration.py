"""Contract tests for the graph-captured prefetch calibration observer."""

import torch

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
    with envs.SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION.override(None):
        assert envs.SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION.get() is False
