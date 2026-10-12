"""r1_run.py: out-root guard, plan freeze before training, DONE reuse, the Stage-2 stop, report and continuation gate.
CPU only; the real-data smoke is opt-in via SGLANG_R1_DATASET, SGLANG_R1_MODEL and SGLANG_R1_CORPUS (with
SGLANG_R1_INTEGRITY and SGLANG_R1_B1_HEAD) and needs CUDA for its cost stage."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_plan  # noqa: E402
import r1_run  # noqa: E402
import r1_stage  # noqa: E402


def _inputs(tmp_path: Path) -> list[str]:
    dataset, corpus = tmp_path / "dataset", tmp_path / "corpus"
    dataset.mkdir()
    corpus.mkdir()
    (dataset / "dataset.json").write_text(json.dumps({"sessions_by_split": {}, "built_sessions": []}))
    (corpus / "manifest.json").write_text(json.dumps({"session_block": {}, "flagged": []}))
    (tmp_path / "integrity.json").write_text(json.dumps({"per_session": []}))
    (tmp_path / "head.pt").write_bytes(b"frozen B1 head")
    return _argv(tmp_path)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    return root


def _main(tmp_path: Path, *stages: str, extra: tuple[str, ...] = ()) -> int:
    root = _root(tmp_path)
    argv = _argv(tmp_path) if (tmp_path / "dataset").exists() else _inputs(tmp_path)
    return r1_run.main([*argv, "--out", str(root / "b1-r1-test"), *extra, "--stages", *stages], out_root=str(root))


def _argv(tmp_path: Path) -> list[str]:
    return ["model", "--dataset", str(tmp_path / "dataset"), "--corpus", str(tmp_path / "corpus"),
            "--integrity", str(tmp_path / "integrity.json"), "--b1-head", str(tmp_path / "head.pt"),
            "--device", "cpu"]


def _out(tmp_path: Path) -> Path:
    return tmp_path / "root" / "b1-r1-test"


def _log(tmp_path: Path) -> list[dict]:
    lines = (_out(tmp_path) / "invocations.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def _cell(recall: float) -> dict:
    return {"threshold": 0.5, "reads": 10, "useful": 5, "wasted": 5, "eligible": 20, "recall": recall,
            "precision": 0.5}


def _stage1_result(*, passed: bool) -> dict:
    recalls = {"gate": 0.30, "gate_floors": 0.32, "F0_noisy_or": 0.40, "hybrid": 0.33}
    tables = {split: {n: {"cap1": _cell(r), "cap2": _cell(r)} for n, r in recalls.items()}
              for split in ("dev", "calib")}
    timing = {"assumptions": {"row_read_ms": 2.2}, "cases": {"nominal": {n: {"on_time_useful": 3} for n in recalls}}}
    gate = {"passed": passed, "reasons": ["recall: met" if passed else "recall: not met"], "values": {}}
    return {"stage1": {**tables, "histograms": {}}, "timing": timing, "stage2_gate": gate,
            "read_records_paths": {}, "label": r1_plan.LABEL}


def _fake_stages(monkeypatch, *, passed: bool) -> list[str]:
    calls: list[str] = []

    def stage1(run):
        calls.append("stage1")
        return r1_run._owned(run.ctx, "stage1", inputs={"fake": "1"}, compute=lambda: _stage1_result(passed=passed))

    def recorder(name):
        def fn(run):
            calls.append(name)
            return {}
        return fn

    monkeypatch.setitem(r1_run._STAGE_FNS, "stage1", stage1)
    for name in ("audit", *r1_run.STAGE2, "cost", "replay"):
        monkeypatch.setitem(r1_run._STAGE_FNS, name, recorder(name))
    return calls


def test_out_outside_the_work_root_or_in_the_b1_run_dir_is_refused_and_not_logged(tmp_path):
    root = _root(tmp_path)
    argv = _inputs(tmp_path)
    assert r1_run.main([*argv, "--out", str(tmp_path / "elsewhere"), "--stages", "plan"], out_root=str(root)) == 2
    bad = root / "b1-run-20261012"
    assert r1_run.main([*argv, "--out", str(bad), "--stages", "plan"], out_root=str(root)) == 2
    assert not (tmp_path / "elsewhere").exists() and not bad.exists()


def test_plan_stage_freezes_a_hashed_plan_and_reuses_it(tmp_path):
    assert _main(tmp_path, "plan") == 0
    out = _out(tmp_path)
    plan, sha = r1_plan.load_plan(str(out / r1_plan.PLAN_NAME))
    assert (out / r1_plan.SHA_NAME).read_text().split()[0] == sha
    assert [a.name for a in plan.arms] == ["F0", "T0", "T1", "T2", "T3", "T4", "T5", "G0"]
    assert plan.timing.contention == {"zero": 1.0, "nominal": 2.0, "adverse": 3.0}
    assert plan.continuation_gate.relative_recall == 0.10 and plan.stage2_gate.cap == 2
    assert sorted(plan.inputs) == sorted(os.path.abspath(p) for p in (
        tmp_path / "dataset" / "dataset.json", tmp_path / "corpus" / "manifest.json", tmp_path / "integrity.json",
        tmp_path / "head.pt"))
    done = json.loads((out / "stages" / "plan" / "DONE.json").read_text())
    assert done["inputs"] == plan.inputs and "result.json" in done["outputs"]
    stamp = (out / r1_plan.PLAN_NAME).stat().st_mtime_ns
    assert _main(tmp_path, "plan") == 0
    assert (out / r1_plan.PLAN_NAME).stat().st_mtime_ns == stamp
    assert [x["exit_code"] for x in _log(tmp_path)] == [0, 0]


def test_a_changed_input_refuses_to_refreeze_and_logs_the_failure(tmp_path):
    assert _main(tmp_path, "plan") == 0
    before = (_out(tmp_path) / r1_plan.PLAN_NAME).read_bytes()
    (tmp_path / "head.pt").write_bytes(b"another head")
    assert _main(tmp_path, "plan") == 1
    assert (_out(tmp_path) / r1_plan.PLAN_NAME).read_bytes() == before
    assert [x["exit_code"] for x in _log(tmp_path)] == [0, 1]


def test_training_is_refused_before_the_plan_is_frozen(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch, passed=True)
    assert _main(tmp_path, "train") == 1
    assert calls == [] and not (_out(tmp_path) / r1_plan.PLAN_NAME).exists()


def test_a_failed_stage2_gate_stops_training_and_the_report_says_so(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch, passed=False)
    assert _main(tmp_path, *r1_run.STAGES) == 0
    assert calls == ["audit", "stage1", "cost", "replay"]
    out = _out(tmp_path)
    results = json.loads((out / r1_run.RESULTS_NAME).read_text())
    assert results["stopped_after_stage1"] and results["label"] == r1_plan.LABEL
    assert results["plan_sha"] == r1_plan.load_plan(str(out / r1_plan.PLAN_NAME))[1]
    assert results["continuation_gate"]["passed"] is False
    assert "stopped after Stage 1" in results["continuation_gate"]["reasons"][0]
    assert results["calibration"]["F0"]["recall_cap2"] == pytest.approx(0.40)
    assert results["attribution"]["ranking"]["status"] == "supported"
    report = (out / r1_run.REPORT_NAME).read_text()
    assert "Stopped after Stage 1" in report and r1_plan.LABEL in report
    assert (out / "stages" / "report" / "DONE.json").exists()


def test_smoke_runs_stage2_even_when_the_gate_fails(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch, passed=False)
    assert _main(tmp_path, *r1_run.STAGES, extra=("--smoke",)) == 0
    assert calls == ["audit", "stage1", *r1_run.STAGE2, "cost", "replay"]
    results = json.loads((_out(tmp_path) / r1_run.RESULTS_NAME).read_text())
    assert results["stage2_forced_by_smoke"] and not results["stopped_after_stage1"]
    assert "have not run" in results["continuation_gate"]["reasons"][0]


def test_owned_stage_is_reused_while_inputs_match_and_reruns_when_they_change(tmp_path):
    ctx = r1_stage.StageCtx(out=str(tmp_path), dataset="d", corpus="c", model_path="m", b1_head="h", device="cpu",
                            smoke=False, plan_sha="p")
    calls = []

    def compute():
        calls.append(1)
        return {"n": len(calls)}

    assert r1_run._owned(ctx, "x", inputs={"a": "1"}, compute=compute) == {"n": 1}
    assert r1_run._owned(ctx, "x", inputs={"a": "1"}, compute=compute) == {"n": 1}
    assert r1_run._owned(ctx, "x", inputs={"a": "2"}, compute=compute) == {"n": 2}
    done = json.loads((tmp_path / "stages" / "x" / "DONE.json").read_text())
    assert done["inputs"] == {"a": "2"} and done["info"]["previous"] == "stale"
    assert done["outputs"]["result.json"] == r1_stage.sha256_file(str(tmp_path / "stages" / "x" / "result.json"))


def test_stage_calib_passes_the_stage1_sha_to_r1_arms(tmp_path, monkeypatch):
    ctx = r1_stage.StageCtx(out=str(tmp_path), dataset="d", corpus="c", model_path="m", b1_head="h", device="cpu",
                            smoke=False, plan_sha="p")
    seen = {}

    def fake_calib(ctx, *, plan, selection, stage1_scorers, stage1_sha):
        seen["sha"] = stage1_sha

    def fake_load(ctx, name):
        return {"selection": {"candidate": "T0"}} if name == "seeds" else {"ok": 1}

    monkeypatch.setattr(r1_run.r1_arms, "stage_calib", fake_calib)
    monkeypatch.setattr(r1_run, "_native_scorers", lambda ctx: {})
    monkeypatch.setattr(r1_run, "_load_result", fake_load)
    monkeypatch.setattr(r1_run, "_result_sha", lambda ctx, name: f"sha-{name}")
    r1_run._stage_calib(SimpleNamespace(ctx=ctx, plan=None))
    assert seen["sha"] == "sha-stage1"


def test_noflag_recall_leaves_out_flagged_sessions():
    def group(session, eligible):
        return SimpleNamespace(key=SimpleNamespace(session=session),
                               labels=SimpleNamespace(eligible_all=np.array(eligible, dtype=np.int16)))

    groups = [group("a", [1, 2]), group("flag", [3, 4]), group("b", [5])]
    ev = {"s": {2: {"reads": [np.array([1, 9]), np.array([3, 4]), np.array([5])]}}}
    assert r1_run._noflag(ev, groups=groups, flagged={"flag"}, cap=2) == {"s": pytest.approx(2 / 3)}


def _gate_parts(*, cand_recall: float, cost_ms: float) -> dict:
    point = {"precision": 0.5, "reads": 100}
    calib = {"scorers": {n: {"2": {"threshold": 0.5, "point": point}} for n in ("gate", "gate_floors", "T2")}}
    dev = {"gate": {"recall": 0.30}, "gate_floors": {"recall": 0.32}, "T2": {"recall": cand_recall}}
    nominal = {"gate": {"on_time_useful": 3}, "gate_floors": {"on_time_useful": 4}, "T2": {"on_time_useful": 9}}
    replay = {"candidate": "T2", "dev": dev, "noflag": {"gate": 0.3, "gate_floors": 0.32, "T2": cand_recall},
              "timing": {"cases": {"nominal": nominal}}}
    seeds = {"selection": {"candidate": "T2", "dev_by_seed": {"T2": {"0": cand_recall, "1": cand_recall, "2": cand_recall}}}}
    return {"seeds": seeds, "calib": calib, "cost": {"gate_ms": cost_ms}, "replay": replay}


def test_continuation_gate_compares_the_candidate_with_the_best_native_control():
    plan = r1_plan.default_plan(dataset="d", corpus="c", b1_head="h", out_dir="o", inputs={})
    ok = r1_run._continuation(plan, **_gate_parts(cand_recall=0.40, cost_ms=0.05))
    assert ok["passed"] and ok["values"]["best_control"] == "gate_floors"
    slow = r1_run._continuation(plan, **_gate_parts(cand_recall=0.40, cost_ms=0.5))
    assert not slow["passed"] and not slow["values"]["arms"]["cost"]
    weak = r1_run._continuation(plan, **_gate_parts(cand_recall=0.33, cost_ms=0.05))
    assert not weak["passed"] and not weak["values"]["arms"]["recall"]


_SMOKE_ENV = ("SGLANG_R1_DATASET", "SGLANG_R1_MODEL", "SGLANG_R1_CORPUS", "SGLANG_R1_INTEGRITY", "SGLANG_R1_B1_HEAD")


@pytest.mark.skipif(any(k not in os.environ for k in _SMOKE_ENV),
                    reason="set " + ", ".join(_SMOKE_ENV) + " to run the real-data smoke")
def test_smoke_runs_every_stage_on_real_data(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("the smoke runs the train, cost and replay stages, which need CUDA")
    root = _root(tmp_path)
    argv = [os.environ["SGLANG_R1_MODEL"], "--dataset", os.environ["SGLANG_R1_DATASET"],
            "--corpus", os.environ["SGLANG_R1_CORPUS"], "--integrity", os.environ["SGLANG_R1_INTEGRITY"],
            "--b1-head", os.environ["SGLANG_R1_B1_HEAD"], "--out", str(root / "b1-r1-smoke"), "--smoke",
            "--device", "cuda"]
    assert r1_run.main(argv, out_root=str(root)) == 0
    out = root / "b1-r1-smoke"
    for name in r1_run.STAGES:
        assert (out / "stages" / name / "DONE.json").exists(), name
    results = json.loads((out / r1_run.RESULTS_NAME).read_text())
    assert results["smoke"] and results["label"] == r1_plan.LABEL
    assert results["cost"]["bf16"]["complete"]["median_ms"] > 0
    assert set(results["timing"]["cases"]) == {"zero", "nominal", "adverse"}
    assert (out / r1_run.REPORT_NAME).read_text().startswith("# R1")
    assert (out / r1_run.SELECTION_NAME).exists()
