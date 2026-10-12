"""r2_plan.py: the T5/G0 specs, freeze/load with its sha256 file, and the continuation gate criteria. CPU only."""

import hashlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402
import r2_plan  # noqa: E402


def _plan(**over):
    r1 = r1_plan.default_plan(dataset="/d", corpus="/c", b1_head="/h", out_dir="/r1", inputs={})
    kw = {"r1_run": "/r1", "r1_plan": r1, "r1_plan_sha": "ab", "inputs": {"/b": "bb", "/a": "aa"},
          "dataset": "/d", "corpus": "/c", "b1_head": "/h", "out_dir": "/mnt/nvme1/nvfp4-work/b1-r2-20261012"}
    kw.update(over)
    return r2_plan.default_plan(**kw)


def test_arms_and_constants():
    plan = _plan()
    arms = {a.name: a for a in plan.arms}
    assert list(arms) == ["T5", "G0"]
    assert arms["T5"].parent == "T3" and arms["G0"].parent is None
    t5 = arms["T5"]
    assert t5.projection and t5.position and t5.adapter_rank == 0 and t5.epochs == 30 and t5.warmup == 3
    assert t5.lr_projection == pytest.approx(1e-4) and t5.route_weight == 0.25
    assert arms["G0"].g0 and arms["G0"].epochs == 60 and arms["G0"].warmup == 5
    assert plan.seeds == (0, 1, 2) and plan.x_dtype == "bfloat16" and plan.x_budget_bytes == 16 << 30
    assert plan.timing.contention == {"zero": 1.0, "nominal": 2.0, "adverse": 3.0}
    assert (plan.groups_per_step, plan.patience, plan.l2_horizon) == (512, 8, 2)
    assert list(plan.inputs) == ["/a", "/b"] and plan.label == r1_plan.LABEL


def test_freeze_roundtrip_and_refusal(tmp_path):
    path = tmp_path / "run" / "r2_plan.json"
    sha = r2_plan.freeze_plan(str(path), _plan())
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (path.parent / "r2_plan.sha256").read_text().split() == [sha, "r2_plan.json"]
    assert r2_plan.freeze_plan(str(path), _plan()) == sha
    plan, loaded = r2_plan.load_plan(str(path))
    assert plan == _plan() and loaded == sha
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        r2_plan.freeze_plan(str(path), _plan(dataset="/other"))
    assert path.read_bytes() == before


def test_load_rejects_a_tampered_plan(tmp_path):
    path = tmp_path / "r2_plan.json"
    r2_plan.freeze_plan(str(path), _plan())
    path.write_bytes(path.read_bytes().replace(b"/d", b"/e"))
    with pytest.raises(ValueError):
        r2_plan.load_plan(str(path))


def _cand(recall=0.60, precision=0.60, reads=1.0):
    return {"recall_cap2": recall, "precision_calib": precision, "reads_calib": reads}


def _gate(**over):
    kw = {"candidate": _cand(), "gate": _cand(0.5), "t3_median": 0.55, "cost_ms": 0.08,
          "seeds": {0: 0.60, 1: 0.56, 2: 0.52}, "noflag": {"candidate": 0.6, "control": 0.5},
          "nominal": {"candidate": 5, "control": 4}}
    kw.update(over)
    return r2_plan.continuation_gate(**kw)


def test_gate_passes_all_criteria():
    out = _gate()
    assert out.passed and all(out.values["arms"].values())


@pytest.mark.parametrize("over, arm", [
    ({"candidate": _cand(recall=0.54), "t3_median": 0.1}, "recall"),
    ({"t3_median": 0.60}, "t3"),
    ({"candidate": _cand(precision=0.49)}, "precision"),
    ({"candidate": _cand(reads=1.06), "gate": _cand(0.5, precision=0.6, reads=1.0)}, "reads"),
    ({"cost_ms": 0.11}, "cost"),
    ({"seeds": {0: 0.60, 1: 0.60, 2: 0.49}}, "seeds"),
    ({"seeds": {0: 0.60, 1: 0.60}}, "seeds"),
    ({"noflag": {"candidate": 0.54, "control": 0.5}}, "noflag"),
    ({"nominal": {"candidate": 4, "control": 4}}, "nominal"),
])
def test_gate_each_criterion_flips_alone(over, arm):
    out = _gate(**over)
    assert not out.passed
    assert [k for k, v in out.values["arms"].items() if not v] == [arm]


def test_gate_bounds_are_inclusive():
    assert _gate(candidate=_cand(reads=1.05), cost_ms=0.1).passed
