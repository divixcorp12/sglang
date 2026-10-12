"""r1_plan.py: the arm ladder, the frozen D16/D17 constants, freeze/load with its sha256 file, and both gates at and
around each threshold. CPU only."""

import hashlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402


def _plan(**over):
    kw = {"dataset": "/d", "corpus": "/c", "b1_head": "/h1_E5_r128_h512_w2.0.pt",
          "out_dir": "/mnt/nvme1/nvfp4-work/b1-r1-20261012", "inputs": {"/b": "bb", "/a": "aa"}}
    kw.update(over)
    return r1_plan.default_plan(**kw)


def test_arm_ladder_follows_d5():
    arms = {a.name: a for a in _plan().arms}
    assert list(arms) == ["F0", "T0", "T1", "T2", "T3", "T4", "T5", "G0"]
    assert [arms[n].parent for n in ("F0", "T0", "T1", "T2", "T3", "T4", "T5", "G0")] == [
        None, None, "T0", "T1", "T2", "T3", "T4", None]
    assert arms["F0"].epochs == 0 and arms["F0"].freeze_encoder
    assert (arms["T0"].objective, arms["T0"].epochs, arms["T0"].warmup) == ("b1_token", 60, 5)
    assert arms["T1"].objective == "block" and arms["T1"].freeze_encoder and arms["T1"].freeze_embeddings
    for n in ("T2", "T3", "T4", "T5", "G0"):
        assert arms[n].objective == "block+route" and arms[n].route_weight == 0.25
    assert arms["T3"].position and arms["T4"].adapter_rank == 16
    assert arms["T5"].projection and arms["T5"].lr_projection == pytest.approx(1e-4)
    assert arms["G0"].g0 and arms["G0"].epochs == 60
    assert all(a.horizon == 1 and a.lr_new == 1e-3 and a.lr_inherited == 1e-4 for a in arms.values())


def test_frozen_constants_and_recipe():
    plan = _plan()
    assert plan.timing.contention == {"zero": 1.0, "nominal": 2.0, "adverse": 3.0}
    assert plan.timing.contention_source == "DSV41_REFERENCE 33.16"
    assert (plan.cost.gate_dtype, plan.cost.head_dtype, plan.cost.aggregator_dtype) == (
        "float32", "bfloat16", "bfloat16")
    assert plan.cost.reference_dtype == "float32" and plan.cost.gate_path == "bf16"
    assert (plan.screen_seed, plan.repeat_seeds, plan.groups_per_step, plan.patience) == (0, (1, 2), 512, 8)
    assert (plan.cap_primary, plan.cap_secondary, plan.mean_reads) == (2, 1, 1.0)
    assert plan.selection_metric.split == "dev" and plan.calibration_rule.split == "calib"
    assert plan.continuation_gate.relative_recall == 0.10 and plan.continuation_gate.extra_reads == 0.05
    assert plan.continuation_gate.cost_ms == 0.1
    assert list(plan.inputs) == ["/a", "/b"]
    assert plan.label == "exploratory on reused data; independent confirmation pending"


def test_freeze_roundtrip_and_sha_file(tmp_path):
    path = tmp_path / "run" / "run_plan.json"
    sha = r1_plan.freeze_plan(str(path), _plan())
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (path.parent / "run_plan.sha256").read_text().split() == [sha, "run_plan.json"]
    assert r1_plan.freeze_plan(str(path), _plan()) == sha
    plan, loaded = r1_plan.load_plan(str(path))
    assert plan == _plan() and loaded == sha


def test_freeze_refuses_a_different_plan(tmp_path):
    path = tmp_path / "run_plan.json"
    r1_plan.freeze_plan(str(path), _plan())
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        r1_plan.freeze_plan(str(path), _plan(dataset="/other"))
    assert path.read_bytes() == before


def test_load_rejects_a_tampered_plan(tmp_path):
    path = tmp_path / "run_plan.json"
    r1_plan.freeze_plan(str(path), _plan())
    path.write_bytes(path.read_bytes().replace(b"/d", b"/e"))
    with pytest.raises(ValueError):
        r1_plan.load_plan(str(path))


def _s2(recall, on_time):
    return {"recall_cap2": recall, "on_time_useful_nominal": on_time}


@pytest.mark.parametrize("f0, passed", [
    (_s2(0.55, 11), True),   # exactly +10% relative is inclusive
    (_s2(0.549, 11), False),
    (_s2(0.60, 10), False),  # on-time useful rows must stay above, not tie
])
def test_stage2_gate_threshold(f0, passed):
    controls = {"gate": _s2(0.40, 50), "gate_floors": _s2(0.50, 10)}
    out = r1_plan.stage2_gate(f0=f0, controls=controls)
    assert out.passed is passed
    assert out.values["best_control"] == "gate_floors"
    assert len(out.reasons) == 2


def _cand(recall=0.60, precision=0.60, reads=1.0):
    return {"recall_cap2": recall, "precision_calib": precision, "reads_calib": reads}


def _cont(**over):
    kw = {"candidate": _cand(), "controls": {"gate": _cand(0.5), "gate_floors": _cand(0.4)}, "cost_ms": 0.08,
          "seeds": {0: 0.60, 1: 0.56, 2: 0.52}, "noflag": {"candidate": 0.6, "control": 0.5},
          "nominal": {"candidate": 5, "control": 4}}
    kw.update(over)
    return r1_plan.continuation_gate(**kw)


def test_continuation_gate_passes_all_arms():
    out = _cont()
    assert out.passed and all(out.values["arms"].values()) and out.values["best_control"] == "gate"


@pytest.mark.parametrize("over, arm", [
    ({"candidate": _cand(recall=0.54)}, "recall"),
    ({"candidate": _cand(precision=0.59)}, "precision"),
    ({"candidate": _cand(reads=1.06)}, "reads"),
    ({"cost_ms": 0.11}, "cost"),
    ({"seeds": {0: 0.60, 1: 0.60, 2: 0.49}}, "seeds"),
    ({"seeds": {0: 0.60, 1: 0.60}}, "seeds"),
    ({"noflag": {"candidate": 0.54, "control": 0.5}}, "noflag"),
    ({"nominal": {"candidate": 4, "control": 4}}, "nominal"),
])
def test_continuation_gate_each_arm_can_fail(over, arm):
    out = _cont(**over)
    assert not out.passed
    assert [k for k, v in out.values["arms"].items() if not v] == [arm]


def test_continuation_gate_bounds_are_inclusive():
    out = _cont(candidate=_cand(reads=1.05), cost_ms=0.1)
    assert out.passed
