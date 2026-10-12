"""r2_eval.py: identical-group check, median selection with parameter ties, per-scorer replay delays. CPU only."""

import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402
import r1_stage1  # noqa: E402
import r2_eval  # noqa: E402
import r2_plan  # noqa: E402

CAP = 2
_Key = namedtuple("_Key", "session seq")


def _plan():
    r1 = r1_plan.default_plan(dataset="/d", corpus="/c", b1_head="/h", out_dir="/r1", inputs={})
    return r2_plan.default_plan(r1_run="/r1", r1_plan=r1, r1_plan_sha="ab", inputs={}, dataset="/d", corpus="/c",
                                b1_head="/h", out_dir="/o")


def _group(seq: int, *, session: str = "s", available=None, eligible=(1, 2)):
    avail = np.ones(8, dtype=bool) if available is None else np.asarray(available, dtype=bool)
    labels = SimpleNamespace(available=avail, eligible_all=np.array(eligible, dtype=np.int16),
                             eligible_addr=np.array(eligible, dtype=np.int16))
    return SimpleNamespace(key=_Key(session, seq), labels=labels)


def _eval(groups, *, reads=None, forwards=None):
    reads = [np.array([1], dtype=np.int64) for _ in groups] if reads is None else reads
    point = {"eligible": sum(len(r.labels.eligible_all) for r in groups),
             "eligible_addr": sum(len(r.labels.eligible_addr) for r in groups),
             "forwards": len(groups) if forwards is None else forwards, "recall": 0.5}
    return {CAP: {"point": point, "reads": reads}}


def test_identical_passes_on_equal_inputs():
    groups = [_group(i) for i in range(3)]
    r2_eval.assert_identical({"a": _eval(groups), "b": _eval(groups)}, groups)
    r2_eval.assert_identical({"a": _eval(groups), "b": _eval(groups)}, {"a": groups, "b": list(groups)})


def test_dropped_group_raises():
    groups = [_group(i) for i in range(3)]
    with pytest.raises(ValueError, match="group keys"):
        r2_eval.assert_identical({"a": _eval(groups), "b": _eval(groups[:2])}, {"a": groups, "b": groups[:2]})
    with pytest.raises(ValueError, match="read lists"):
        r2_eval.assert_identical({"a": _eval(groups, reads=[np.array([1])] * 2)}, groups)


def test_changed_skip_mask_raises():
    groups = [_group(i) for i in range(2)]
    masked = [_group(0), _group(1, available=[1, 0, 1, 1, 1, 1, 1, 1])]
    with pytest.raises(ValueError, match="skip mask"):
        r2_eval.assert_identical({"a": _eval(groups), "b": _eval(masked)}, {"a": groups, "b": masked})
    with pytest.raises(ValueError, match="skipped row"):
        r2_eval.assert_identical({"a": _eval(masked, reads=[np.array([1]), np.array([1])])}, masked)


def test_changed_denominator_raises():
    groups = [_group(i) for i in range(2)]
    other = [_group(i, eligible=(1,)) for i in range(2)]
    with pytest.raises(ValueError, match="eligible denominators"):
        r2_eval.assert_identical({"a": _eval(other)}, groups)


def _arm(recalls: list[float], *, params: int, best: int = 0) -> dict:
    return {"curve": [{"epoch": i, "recall": r} for i, r in enumerate(recalls)], "best_epoch": best,
            "params": params, "ckpt_path": f"/c/{params}.pt", "ckpt_sha": f"sha{params}"}


def _train(t5: dict, g0: dict) -> dict:
    return {"arms": {"T5": t5, "G0": g0}}


def _seeds(t3: float) -> dict:
    return {"selection": {"median_by_arm": {"T3": t3}}}


def test_select_takes_median_not_best_seed():
    t5 = {"0": _arm([0.50], params=10), "1": _arm([0.50], params=10), "2": _arm([0.90], params=10)}
    g0 = {"0": _arm([0.55], params=20), "1": _arm([0.55], params=20), "2": _arm([0.40], params=20)}
    out = r2_eval.select(_train(t5, g0), r1_seeds=_seeds(0.45))
    assert out["median_by_arm"] == {"T5": 0.5, "G0": 0.55}
    assert out["winner"] == "G0" and out["t3_median"] == 0.45
    assert out["candidate"] == {"path": "/c/20.pt", "sha": "sha20"}


def test_select_uses_best_epoch_and_breaks_ties_by_params():
    t5 = {s: _arm([0.1, 0.6, 0.2], params=10, best=1) for s in "012"}
    g0 = {s: _arm([0.6], params=20) for s in "012"}
    out = r2_eval.select(_train(t5, g0), r1_seeds=_seeds(0.5))
    assert out["median_by_arm"]["T5"] == pytest.approx(0.6) and out["winner"] == "T5"
    assert out["candidate"]["sha"] == "sha10"


def test_select_accepts_int_seed_keys():
    t5 = {s: _arm([0.5], params=10) for s in (0, 1, 2)}
    g0 = {s: _arm([0.6], params=20) for s in (0, 1, 2)}
    out = r2_eval.select(_train(t5, g0), r1_seeds=_seeds(0.45))
    assert out["winner"] == "G0" and out["candidate"]["sha"] == "sha20"


def test_delay_per_scorer():
    r1_cost = {"gate_only_ms": 0.01, "gate_ms": 0.2}
    delays = r2_eval.scorer_delays(r1_cost=r1_cost, arm_costs={"T5": {"complete_ms": 0.3}, "G0": {"complete_ms": 0.4}})
    pick = {n: r1_stage1._delay_for(delays, name=n) for n in ("gate", "T3", "F0", "T5", "G0")}
    assert pick == {"gate": 0.01, "T3": 0.2, "F0": 0.2, "T5": 0.3, "G0": 0.4}
    l2 = r2_eval._l2_delays(delays, names=["G0", "T3"])
    assert (l2["G0_L2"], l2["T3_L2"], l2["gate"]) == (0.4, 0.2, 0.01)


def test_replay_passes_delays_and_computes_noflag(monkeypatch):
    seen = {}

    def fake_timing(evals, *, groups, plan, windows, scorer_delay_ms):
        seen.update(delay=scorer_delay_ms, cap=plan.stage2_gate.cap, split=plan.stage2_gate.split)
        return {"cases": {}}

    monkeypatch.setattr(r1_stage1, "_timing", fake_timing)
    plan = _plan()
    groups = [_group(0, session="ok", eligible=(1, 2)), _group(1, session="bad", eligible=(1, 2))]
    reads = [np.array([1], dtype=np.int64), np.array([1, 2], dtype=np.int64)]
    ev = {"a": {CAP: {"point": {"recall": 0.5}, "reads": reads}}}
    delays = {"a": 0.3, "*": 0.1}
    out = r2_eval.replay({"dev": ev, "calib": ev}, groups={"dev": groups, "calib": groups}, plan=plan, windows={},
                         delays=delays, flagged={"bad"})
    assert seen == {"delay": delays, "cap": 2, "split": "dev"}
    assert out["noflag"] == {"a": 0.5} and out["dev"] == {"a": {"recall": 0.5}}
    assert out["scorer_delay_ms"] == delays


def test_train_split_is_refused():
    with pytest.raises(ValueError, match="neither"):
        r2_eval.evaluate([], [], plan=_plan(), split="train")
