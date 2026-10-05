"""The DSpark draft's resident-set file: calibration from a routes probe, and a loader that refuses bad files."""

import json

import pytest

from sglang.srt.layers.moe.cpu_experts import draft_resident
from sglang.srt.layers.moe.cpu_experts.draft_resident import (
    load_resident_set,
    resident_for,
    top_n_resident_set,
    write_resident_set,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _routes():
    return [
        json.dumps({"layer": 0, "ids": [[5, 1, 2], [5, 1, 3]]}),
        json.dumps({"layer": 1, "ids": [[7, 8, 9], [9, 8, 4]]}),
        json.dumps({"layer": 0, "ids": [[5, 2, 6], [2, 0, 1]]}),
    ]


def test_top_n_takes_the_most_routed_ids_per_layer_ties_to_the_lower_id():
    # layer 0 counts: 5:3, 1:3, 2:3, 3:1, 6:1, 0:1; layer 1: 8:2, 9:2, 7:1, 4:1
    assert top_n_resident_set(_routes(), 2) == {0: [1, 2], 1: [8, 9]}


def test_a_written_file_loads_back_as_frozensets(tmp_path):
    path = tmp_path / "resident.json"
    write_resident_set(str(path), {0: [1, 2], 1: [8, 9]}, n=2, source="routes.jsonl")
    assert json.loads(path.read_text())["version"] == 1
    assert load_resident_set(str(path)) == {0: frozenset({1, 2}), 1: frozenset({8, 9})}


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"version": 2, "n": 1, "source": "", "stages": {"0": [1]}}),
        json.dumps({"version": 1, "n": 1, "source": "", "stages": {"0": [-1]}}),
        json.dumps({"version": 1, "n": 2, "source": "", "stages": {"0": [1, 1]}}),
        json.dumps({"version": 1, "n": 1, "source": "", "stages": {"x": [1]}}),
        json.dumps({"version": 1, "n": 1, "source": "", "stages": {"0": [1, 2]}}),
        json.dumps({"version": 1, "source": "", "stages": {"0": [1]}}),
    ],
)
def test_a_malformed_file_is_refused_naming_the_path(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="bad.json"):
        load_resident_set(str(path))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nope.json"):
        load_resident_set(str(tmp_path / "nope.json"))


def test_resident_for_follows_the_env_and_refuses_an_unlisted_stage(tmp_path):
    from sglang.srt.environ import envs

    path = tmp_path / "resident.json"
    write_resident_set(str(path), {0: [3]}, n=1, source="")
    draft_resident._cache.clear()
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(""):
        assert resident_for(0) == frozenset()
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        assert resident_for(0) == frozenset({3})
        with pytest.raises(ValueError, match="stage 1"):
            resident_for(1)
