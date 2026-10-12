"""b1_run.py: stage resume, the locked-test guard, a small end-to-end run on a synthetic dataset, smoke inputs.
CPU only; the real-corpus smoke is opt-in via SGLANG_B1_CORPUS (with SGLANG_B1_MODEL and SGLANG_B1_INTEGRITY)."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import b1_run  # noqa: E402
from test_b1_rows import _group  # noqa: E402

DIM = 8
ROWS = 6
SPLIT_SESSIONS = {"train": 8, "dev": 2, "calib": 2, "locked_test": 2}
HORIZONS = (1, 2)


def _small_config() -> b1_run.RunConfig:
    grid = {"E2": [{"variant": "E2", "pca_rank": 4, "hidden": 16, "hard_neg_weight": 1.0}],
            "E5": [{"variant": "E5", "pca_rank": 4, "hidden": 16, "hard_neg_weight": 2.0}]}
    tcfg = b1_run.b1_train.TrainConfig(epochs=2, lr=1e-2, batch_rows=8, patience=1, seed=0, device="cpu")
    return b1_run.RunConfig(horizons=HORIZONS, grids=grid, train=tcfg, bootstrap_n=50, cost_runs=2, cost_warmup=1,
                            floors={1: 0.5, 2: 0.5})


def _write_dataset(root: Path) -> Path:
    rng = np.random.default_rng(0)
    sessions: dict[str, list[str]] = {}
    for split, count in SPLIT_SESSIONS.items():
        sessions[split] = [f"{split[:2]}{i}" for i in range(count)]
        for name in sessions[split]:
            words = (rng.standard_normal((ROWS, DIM)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            b1_rows.write_x(root, split=split, session=name, x=words)
            for h in HORIZONS:
                groups = [_group(seq=i, n=1, x_offset=i, session=name, split=split, horizon=h) for i in range(ROWS)]
                b1_rows.write_groups(root, split=split, horizon=h, session=name, groups=groups)
    info = {"counts": {}, "warmup_forwards": 0, "dropped_cells": 0, "invalid_horizons": 0,
            "sessions_by_split": sessions, "built_sessions": sorted(s for v in sessions.values() for s in v),
            "flagged": [], "manifest_sha256": ""}
    (root / "dataset.json").write_text(json.dumps(info))
    return root


def _args(tmp_path: Path, dataset: Path, *extra: str):
    return b1_run._parse(["model", "--corpus", "c", "--integrity", "i", "--out", str(tmp_path / "out"),
                          "--dataset", str(dataset), "--device", "cpu", "--allow-any-out", *extra])


def _ctx(tmp_path: Path, dataset: Path) -> b1_run._Ctx:
    return b1_run._Ctx(model_path="model", corpus="c", integrity="i", out=str(tmp_path / "out"),
                       dataset_dir=str(dataset), cfg=_small_config(), smoke=False, device="cpu")


def test_stages_resume_at_the_first_missing_marker(tmp_path):
    calls = []

    def stage(name):
        def fn(ctx):
            calls.append(name)
            return {"name": name}
        return fn

    fns = {n: stage(n) for n in ("a", "b", "c")}
    ctx = SimpleNamespace(out=str(tmp_path))
    assert b1_run._run_stages(ctx=ctx, fns=fns, selected=["a", "b", "c"]) == {"a": True, "b": True, "c": True}
    (tmp_path / "stages" / "b" / "DONE.json").unlink()
    (tmp_path / "stages" / "c" / "DONE.json").unlink()
    calls.clear()
    assert b1_run._run_stages(ctx=ctx, fns=fns, selected=["a", "b", "c"]) == {"a": False, "b": True, "c": True}
    assert calls == ["b", "c"]


def test_a_stage_is_refused_when_an_earlier_one_has_no_marker(tmp_path):
    fns = {"a": lambda ctx: {}, "b": lambda ctx: {}}
    with pytest.raises(RuntimeError, match="'a'"):
        b1_run._run_stages(ctx=SimpleNamespace(out=str(tmp_path)), fns=fns, selected=["b"])


def test_end_to_end_on_a_synthetic_dataset_resumes_and_records_the_frozen_sha(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset")
    args, cfg = _args(tmp_path, dataset), _small_config()
    out = tmp_path / "out"
    upto_locked = list(b1_run.STAGES[:b1_run.STAGES.index("cost")])
    b1_run.run(_args(tmp_path, dataset, "--stages", *upto_locked), cfg=cfg)
    with pytest.raises(RuntimeError, match="cc-gpu.lock"):
        b1_run.run(_args(tmp_path, dataset, "--stages", "cost"), cfg=cfg)
    assert not (out / "stages" / "cost" / "DONE.json").exists()
    # Stand-in for a GPU cost run; the CPU check has no CUDA.
    fake_cost = {"bytes_bf16": 1000, "median_ms": 0.05}
    (out / "stages" / "cost").mkdir(parents=True, exist_ok=True)
    (out / "stages" / "cost" / "DONE.json").write_text(json.dumps(fake_cost))
    results = b1_run.run(_args(tmp_path, dataset, "--stages", "report"), cfg=cfg)

    sha = hashlib.sha256((out / "selection.json").read_bytes()).hexdigest()
    assert results["frozen_sha256"] == sha
    assert set(results["horizons"]) == {"1", "2"}
    tables = results["horizons"]["1"]["tables"]
    assert set(tables) == {"gate", "gate_floors", "prev", "pop", "E2", "E5"}
    assert "gate_floors" not in results["horizons"]["2"]["tables"]
    pooled = {name: t["cells"]["reads=1"]["pooled"] for name, t in tables.items()}
    assert len({(c["forwards"], c["eligible"], c["never_candidates"]) for c in pooled.values()}) == 1
    assert pooled["gate"]["eligible"] == SPLIT_SESSIONS["locked_test"] * ROWS
    assert set(results["learning_curve"]["E2"]) == {"15", "30", "60"}
    assert results["cost"] == fake_cost
    assert isinstance(results["verdict"]["advance"], bool)
    assert (out / "stages" / "report" / "REPORT.md").read_text().startswith("# B1 learned predictor")

    train_marker = out / "stages" / "train" / "DONE.json"
    stamp = train_marker.stat().st_mtime_ns
    (out / "stages" / "report" / "DONE.json").unlink()
    again = b1_run.run(args, cfg=cfg)
    assert train_marker.stat().st_mtime_ns == stamp
    assert again["frozen_sha256"] == sha


def test_locked_stage_refuses_without_a_frozen_selection(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset")
    b1_run.run(_args(tmp_path, dataset, "--stages", *b1_run.STAGES[:6]), cfg=_small_config())
    assert not (tmp_path / "out" / "selection.json").exists()
    with pytest.raises(PermissionError):
        b1_run._stage_locked(_ctx(tmp_path, dataset))
    with pytest.raises(RuntimeError, match="freeze"):
        b1_run.run(_args(tmp_path, dataset, "--stages", "locked"), cfg=_small_config())


def test_out_off_nvme_is_refused_without_allow_any_out(tmp_path):
    with pytest.raises(SystemExit):
        b1_run._parse(["model", "--corpus", "c", "--integrity", "i", "--out", str(tmp_path / "out")])


def test_smoke_takes_only_train_split_sessions_and_scratch_rids_name_only_them(tmp_path):
    manifest = {"split": {"train": [f"t{i}" for i in range(7)], "dev": ["d0"], "locked_test": ["l0", "l1"]},
                "session_block": {s: "b000" for s in [f"t{i}" for i in range(7)] + ["d0", "l0", "l1"]}}
    picked = b1_run._pick_smoke_sessions(manifest)
    assert {k: len(v) for k, v in picked.items()} == b1_run.SMOKE_SESSIONS
    assert {s for v in picked.values() for s in v} <= set(manifest["split"]["train"])

    src = tmp_path / "src"
    src.mkdir()
    (src / "trace.jsonl").write_text("{}")
    (src / "rids.json").write_text(json.dumps({"r0": {"session_id": "t0"}, "r1": {"session_id": "l0"}}))
    assert b1_run._scratch_block(str(src), str(tmp_path / "dst"), sessions={"t0"})
    assert json.loads((tmp_path / "dst" / "rids.json").read_text()) == {"r0": {"session_id": "t0"}}
    assert (tmp_path / "dst" / "trace.jsonl").is_symlink()
    assert not b1_run._scratch_block(str(src), str(tmp_path / "none"), sessions={"zz"})


def test_smoke_needs_five_train_sessions():
    with pytest.raises(ValueError, match="needs 5"):
        b1_run._pick_smoke_sessions({"split": {"train": ["a", "b"]}, "session_block": {"a": "b", "b": "b"}})


@pytest.mark.skipif("SGLANG_B1_CORPUS" not in os.environ, reason="set SGLANG_B1_CORPUS to run the real-corpus smoke")
def test_smoke_runs_every_stage_on_real_training_sessions(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("the smoke runs the cost stage, which needs CUDA")
    corpus = os.environ["SGLANG_B1_CORPUS"]
    argv = [os.environ["SGLANG_B1_MODEL"], "--corpus", corpus, "--integrity", os.environ["SGLANG_B1_INTEGRITY"],
            "--out", str(tmp_path / "out"), "--smoke", "--device", "cuda", "--allow-any-out"]
    assert b1_run.main(argv) == 0
    results = json.loads((tmp_path / "out" / "stages" / "report" / "results.json").read_text())
    assert results["smoke"] and results["frozen_sha256"]
    manifest = json.loads((tmp_path / "out" / "smoke" / "manifest.json").read_text())
    real = json.loads(Path(corpus, "manifest.json").read_text())
    used = {s for sessions in manifest["split"].values() for s in sessions}
    assert used <= set(real["split"]["train"])
