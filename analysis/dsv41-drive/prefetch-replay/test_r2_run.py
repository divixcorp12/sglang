"""r2_run.py: out-root guard, plan freeze over the R1 hashes, refusal before the plan, DONE reuse, the budget exit,
the report and continuation gate, and an R1 dir left byte-identical. CPU only, on a fake R1 run dir; the real-data
smoke is opt-in via SGLANG_R2_R1_RUN, SGLANG_R2_DATASET, SGLANG_R2_MODEL, SGLANG_R2_CORPUS and SGLANG_R2_B1_HEAD
and needs CUDA."""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402
import r1_stage  # noqa: E402
import r2_arms  # noqa: E402
import r2_plan  # noqa: E402
import r2_run  # noqa: E402

_OUT = "b1-r2-test"


# ---- fixtures ----------------------------------------------------------------------------------------------------


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _ckpt(path: Path, body: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body.encode())
    return {"path": str(path), "sha": r1_stage.sha256_file(str(path))}


def _fake_r1(tmp_path: Path) -> Path:
    """A finished R1 run: frozen run_plan.json, T3/F0 seed-0 and L+2 T3 checkpoints, and the stage results R2 reads."""
    r1 = tmp_path / "r1"
    stages = r1 / "stages"
    t3, f0 = _ckpt(stages / "seeds" / "ckpt" / "T3.pt", "T3 seed 0"), _ckpt(stages / "seeds" / "ckpt" / "F0.pt", "F0")
    t3_l2 = _ckpt(stages / "l2" / "ckpt" / "T3.pt", "T3 at L+2")
    plan = r1_plan.default_plan(dataset="d", corpus="c", b1_head="h", out_dir=str(r1), inputs={})
    r1_plan.freeze_plan(str(r1 / r1_plan.PLAN_NAME), plan)
    selection = {"candidate": "T3", "median_by_arm": {"T3": 0.36}, "ckpts": {"T3": t3, "F0": f0}}
    _write(stages / "seeds" / "result.json", {"selection": selection, "label": r1_plan.LABEL})
    _write(r1 / "selection.json", selection)
    _write(stages / "l2" / "result.json", {"arms": {"T3": {"ckpt_path": t3_l2["path"], "ckpt_sha": t3_l2["sha"]}}})
    _write(stages / "cost" / "result.json", {"gate_ms": 0.2, "gate_only_ms": 0.01, "layer": 3})
    _write(stages / "calib" / "result.json", {"scorers": {}})
    _write(stages / "audit" / "exclusions.json", {"cells": []})
    return r1


def _inputs(tmp_path: Path) -> None:
    (tmp_path / "dataset").mkdir()
    (tmp_path / "corpus").mkdir()
    (tmp_path / "head.pt").write_bytes(b"frozen B1 head")
    _fake_r1(tmp_path)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    return root


def _argv(tmp_path: Path, *, r1: Path | None = None) -> list[str]:
    return ["model", "--r1-run", str(r1 or tmp_path / "r1"), "--dataset", str(tmp_path / "dataset"),
            "--corpus", str(tmp_path / "corpus"), "--b1-head", str(tmp_path / "head.pt"), "--device", "cpu"]


def _main(tmp_path: Path, *stages: str) -> int:
    if not (tmp_path / "r1").exists():
        _inputs(tmp_path)
    root = _root(tmp_path)
    return r2_run.main([*_argv(tmp_path), "--out", str(root / _OUT), "--stages", *stages], out_root=str(root))


def _out(tmp_path: Path) -> Path:
    return tmp_path / "root" / _OUT


def _log(tmp_path: Path) -> list[dict]:
    return [json.loads(x) for x in (_out(tmp_path) / "invocations.jsonl").read_text().splitlines()]


def _tree(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


# ---- synthetic stage results -------------------------------------------------------------------------------------


def _arm(recall: float) -> dict:
    return {"curve": [{"epoch": -1, "recall": 0.30}, {"epoch": 0, "recall": recall}], "best_epoch": 0, "params": 10,
            "ckpt_path": "ckpt.pt", "ckpt_sha": "c" * 64, "parent_sha": ""}


def _train() -> dict:
    recalls = {"T5": {"0": 0.40, "1": 0.41, "2": 0.39}, "G0": {"0": 0.35, "1": 0.34, "2": 0.36}}
    return {"arms": {a: {s: _arm(r) for s, r in by.items()} for a, by in recalls.items()}, "dev_recall": recalls,
            "label": r1_plan.LABEL}


def _point(recall: float) -> dict:
    return {"threshold": 0.5, "recall": recall, "precision": 0.5, "reads": 100}


def _select() -> dict:
    return {"winner": "T5", "median_by_arm": {"T5": 0.40, "G0": 0.35}, "t3_median": 0.36,
            "dev_by_seed": _train()["dev_recall"], "candidate": {"path": "ckpt.pt", "sha": "c" * 64},
            "comparison": {"cap2": {n: _point(r) for n, r in (("T5", 0.40), ("gate", 0.30), ("T3", 0.36))}},
            "label": r1_plan.LABEL}


def _calib() -> dict:
    return {"scorers": {n: {"1": _point(0.2), "2": _point(0.3)} for n in ("T5", "gate", "T3", "F0")},
            "label": r1_plan.LABEL}


def _cost(*, increment_ms: float) -> dict:
    t = {"median_ms": 0.05, "p90_ms": 0.06, "p99_ms": 0.07, "runs": 50}
    row = {"head_only": t, "complete": t, "param_bytes": 1, "buffer_bytes": 2}
    arms = {"T5": {"bf16": row, "fp32": row, "increment_ms": increment_ms, "complete_ms": 0.2},
            "G0": {"bf16": {**row, "correction": t}, "fp32": {**row, "correction": t}, "increment_ms": 0.05,
                   "complete_ms": 0.3}}
    return {"arms": arms, "layer": 3, "label": r1_plan.LABEL}


def _replay() -> dict:
    nominal = {"T5": {"on_time_useful": 9}, "gate": {"on_time_useful": 3}, "T3": {"on_time_useful": 6}}
    return {"dev": {"T5": {"recall": 0.40}, "gate": {"recall": 0.30}, "T3": {"recall": 0.36}},
            "noflag": {"T5": 0.40, "gate": 0.30, "T3": 0.36},
            "timing": {"assumptions": {"row_read_ms": 2.2}, "cases": {"nominal": nominal}},
            "scorer_delay_ms": {"gate": 0.01, "*": 0.2, "T5": 0.2}, "label": r1_plan.LABEL}


def _fake_stages(monkeypatch, *, increment_ms: float = 0.05, tag: str = "1") -> list[str]:
    """Every stage after plan writes its synthetic result through r2_run._owned; l2 only records the call."""
    calls: list[str] = []
    results = {"train": _train, "select": _select, "calib": _calib, "cost": lambda: _cost(increment_ms=increment_ms),
               "replay": _replay}

    def owned(name):
        def fn(run):
            calls.append(name)
            return r2_run._owned(run.ctx, name, inputs={"fake": tag}, compute=results[name])
        return fn

    def l2(run):
        calls.append("l2")
        return {}

    for name in results:
        monkeypatch.setitem(r2_run._STAGE_FNS, name, owned(name))
    monkeypatch.setitem(r2_run._STAGE_FNS, "l2", l2)
    return calls


# ---- tests -------------------------------------------------------------------------------------------------------


def test_out_outside_the_root_in_protected_runs_or_inside_the_r1_run_is_refused_and_not_logged(tmp_path):
    _inputs(tmp_path)
    root = _root(tmp_path)
    r1_in_root = root / "r1-src"
    r1_in_root.mkdir()
    cases = [(tmp_path / "elsewhere", None), (root / "b1-run-20261012", None), (root / "b1-r1-20261012", None),
             (root / "b1-r1-20261012" / "r2", None), (r1_in_root, r1_in_root), (r1_in_root / "r2", r1_in_root)]
    for out, r1 in cases:
        argv = [*_argv(tmp_path, r1=r1), "--out", str(out), "--stages", "plan"]
        assert r2_run.main(argv, out_root=str(root)) == 2, out
    assert not (tmp_path / "elsewhere").exists() and not (root / "b1-r1-20261012").exists()
    assert not (root / "b1-run-20261012").exists() and os.listdir(r1_in_root) == []


def test_plan_stage_freezes_the_r1_hashes_and_reuses_the_plan(tmp_path):
    assert _main(tmp_path, "plan") == 0
    out = _out(tmp_path)
    plan, sha = r2_plan.load_plan(str(out / r2_plan.PLAN_NAME))
    assert (out / r2_plan.SHA_NAME).read_text().split()[0] == sha
    assert plan.r1_run == os.path.realpath(tmp_path / "r1") and plan.label == r1_plan.LABEL
    assert [a.name for a in plan.arms] == ["T5", "G0"] and plan.seeds == (0, 1, 2)
    assert set(plan.inputs) == {*r2_run._R1_FILES, *r2_run._R1_CKPTS, "b1_head"}
    assert plan.inputs["t3_seed0"] == r1_stage.sha256_file(str(tmp_path / "r1" / "stages" / "seeds" / "ckpt" / "T3.pt"))
    assert plan.r1_plan_sha == r1_plan.load_plan(str(tmp_path / "r1" / r1_plan.PLAN_NAME))[1]
    stamp = (out / r2_plan.PLAN_NAME).stat().st_mtime_ns
    assert _main(tmp_path, "plan") == 0
    assert (out / r2_plan.PLAN_NAME).stat().st_mtime_ns == stamp
    assert [x["exit_code"] for x in _log(tmp_path)] == [0, 0]


def test_a_changed_r1_input_refuses_to_refreeze_and_refuses_later_stages(tmp_path, monkeypatch):
    assert _main(tmp_path, "plan") == 0
    before = (_out(tmp_path) / r2_plan.PLAN_NAME).read_bytes()
    _write(tmp_path / "r1" / "stages" / "cost" / "result.json", {"gate_ms": 0.3, "gate_only_ms": 0.01, "layer": 3})
    assert _main(tmp_path, "plan") == 1
    assert (_out(tmp_path) / r2_plan.PLAN_NAME).read_bytes() == before
    calls = _fake_stages(monkeypatch)
    assert _main(tmp_path, "train") == 1 and calls == []
    assert [x["exit_code"] for x in _log(tmp_path)] == [0, 1, 1]


def test_a_checkpoint_that_differs_from_r1s_record_is_refused(tmp_path):
    _inputs(tmp_path)
    (tmp_path / "r1" / "stages" / "seeds" / "ckpt" / "T3.pt").write_bytes(b"tampered")
    assert _main(tmp_path, "plan") == 1
    assert not (_out(tmp_path) / r2_plan.PLAN_NAME).exists()


def test_training_is_refused_before_the_plan_is_frozen(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    assert _main(tmp_path, "train") == 1
    assert calls == [] and not (_out(tmp_path) / r2_plan.PLAN_NAME).exists()


def test_the_x_budget_refusal_is_a_clean_nonzero_exit_with_its_message(tmp_path, monkeypatch, capsys):
    assert _main(tmp_path, "plan") == 0

    def stage_train(ctx, *, plan, **_):
        r2_arms.check_x_budget(n_groups=1 << 30, plan=plan, free_bytes=1 << 50)

    monkeypatch.setattr(r2_run.r2_arms, "stage_train", stage_train)
    assert _main(tmp_path, "train") == r2_run.BUDGET_EXIT
    err = capsys.readouterr().err
    assert "budget" in err and "Traceback" not in err
    assert _log(tmp_path)[-1]["exit_code"] == r2_run.BUDGET_EXIT


def test_the_train_stage_passes_r1s_t3_seed0_as_t5s_parent(tmp_path, monkeypatch):
    assert _main(tmp_path, "plan") == 0
    seen = {}

    def stage_train(ctx, **kw):
        seen.update(kw, plan_sha=ctx.plan_sha)
        return {}

    monkeypatch.setattr(r2_run.r2_arms, "stage_train", stage_train)
    assert _main(tmp_path, "train") == 0
    t3 = tmp_path / "r1" / "stages" / "seeds" / "ckpt" / "T3.pt"
    assert seen["parents"] == {"T5": (str(t3), r1_stage.sha256_file(str(t3)))}
    assert seen["stage"] == "train" and seen["horizon"] == 1 and seen["seeds"] == (0, 1, 2)
    assert seen["arms"] == ("T5", "G0") and seen["plan_sha"] == r2_plan.load_plan(
        str(_out(tmp_path) / r2_plan.PLAN_NAME))[1]


def test_a_full_run_reports_the_gate_and_leaves_the_r1_dir_byte_identical(tmp_path, monkeypatch):
    _inputs(tmp_path)
    before = _tree(tmp_path / "r1")
    calls = _fake_stages(monkeypatch)
    assert _main(tmp_path, *r2_run.STAGES) == 0
    assert calls == ["train", "select", "calib", "cost", "replay", "l2"]
    assert _tree(tmp_path / "r1") == before
    out = _out(tmp_path)
    results = json.loads((out / r2_run.RESULTS_NAME).read_text())
    assert results["label"] == r1_plan.LABEL
    assert results["plan_sha"] == r2_plan.load_plan(str(out / r2_plan.PLAN_NAME))[1]
    assert results["continuation_gate"]["passed"] is True and results["adds_over_t3"] is True
    assert results["selection"]["winner"] == "T5" and results["selection"]["median_dev_recall"]["T3"] == 0.36
    assert results["cost"]["G0"]["bf16"]["correction"]["param_bytes"] == 1
    assert results["arms"]["T5"]["1"]["dev_recall_cap2"] == 0.41
    report = (out / r2_run.REPORT_NAME).read_text()
    assert r1_plan.LABEL in report and "Outcome: PASS" in report
    assert (out / "stages" / "report" / "DONE.json").exists()


def test_the_report_is_reused_on_matching_inputs_and_rerun_when_a_stage_changes(tmp_path, monkeypatch):
    _fake_stages(monkeypatch)
    assert _main(tmp_path, *r2_run.STAGES) == 0
    report = _out(tmp_path) / r2_run.REPORT_NAME
    stamp = report.stat().st_mtime_ns
    assert _main(tmp_path, "report") == 0
    assert report.stat().st_mtime_ns == stamp
    _fake_stages(monkeypatch, increment_ms=0.5, tag="2")
    assert _main(tmp_path, "cost", "report") == 0
    results = json.loads((_out(tmp_path) / r2_run.RESULTS_NAME).read_text())
    assert results["continuation_gate"]["passed"] is False and results["adds_over_t3"] is False
    assert not results["continuation_gate"]["values"]["arms"]["cost"]


def test_missing_stages_leave_the_gate_not_evaluated_and_numbers_pending(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    assert _main(tmp_path, "plan", "train", "report") == 0
    assert calls == ["train"]
    results = json.loads((_out(tmp_path) / r2_run.RESULTS_NAME).read_text())
    assert not results["continuation_gate"]["passed"] and results["adds_over_t3"] is None
    assert "have not run" in results["continuation_gate"]["reasons"][0]
    assert "adds anything over T3: pending" in (_out(tmp_path) / r2_run.REPORT_NAME).read_text()


def test_owned_stage_is_reused_while_inputs_match_and_reruns_when_they_change(tmp_path):
    ctx = r1_stage.StageCtx(out=str(tmp_path), dataset="d", corpus="c", model_path="m", b1_head="h", device="cpu",
                            smoke=False, plan_sha="p")
    calls = []

    def compute():
        calls.append(1)
        return {"n": len(calls)}

    assert r2_run._owned(ctx, "x", inputs={"a": "1"}, compute=compute) == {"n": 1}
    assert r2_run._owned(ctx, "x", inputs={"a": "1"}, compute=compute) == {"n": 1}
    assert r2_run._owned(ctx, "x", inputs={"a": "2"}, compute=compute) == {"n": 2}
    done = json.loads((tmp_path / "stages" / "x" / "DONE.json").read_text())
    assert done["inputs"] == {"a": "2"} and done["info"]["previous"] == "stale"


_SMOKE_ENV = ("SGLANG_R2_R1_RUN", "SGLANG_R2_DATASET", "SGLANG_R2_MODEL", "SGLANG_R2_CORPUS", "SGLANG_R2_B1_HEAD")


@pytest.mark.skipif(any(k not in os.environ for k in _SMOKE_ENV),
                    reason="set " + ", ".join(_SMOKE_ENV) + " to run the real-data smoke")
def test_smoke_runs_every_stage_on_real_data(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("the smoke trains and times T5 and G0, which needs CUDA")
    r1 = Path(os.environ["SGLANG_R2_R1_RUN"])
    guarded = [r1 / r1_plan.PLAN_NAME, r1 / "selection.json", r1 / "stages" / "seeds" / "result.json"]
    before = {p: r1_stage.sha256_file(str(p)) for p in guarded}
    root = _root(tmp_path)
    argv = [os.environ["SGLANG_R2_MODEL"], "--r1-run", str(r1), "--dataset", os.environ["SGLANG_R2_DATASET"],
            "--corpus", os.environ["SGLANG_R2_CORPUS"], "--b1-head", os.environ["SGLANG_R2_B1_HEAD"],
            "--out", str(root / "b1-r2-smoke"), "--smoke", "--device", "cuda"]
    assert r2_run.main(argv, out_root=str(root)) == 0
    out = root / "b1-r2-smoke"
    for name in (*r2_run.STAGES, r2_run.L2_TABLE):
        assert (out / "stages" / name / "DONE.json").exists(), name
    results = json.loads((out / r2_run.RESULTS_NAME).read_text())
    assert results["smoke"] and results["label"] == r1_plan.LABEL
    assert results["cost"]["G0"]["bf16"]["correction"]["median_ms"] > 0
    assert set(results["timing"]["cases"]) == {"zero", "nominal", "adverse"}
    assert (out / r2_run.SELECTION_NAME).exists()
    assert {p: r1_stage.sha256_file(str(p)) for p in guarded} == before
