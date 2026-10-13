"""l2_cap1_eval: the --out guard, --cost parsing and the block-score cache."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l2_cap1_eval  # noqa: E402


def _root(tmp_path: Path) -> tuple[str, str]:
    root = tmp_path / "work"
    run_dir = root / "b1-r1-20261012"
    run_dir.mkdir(parents=True)
    return str(root), str(run_dir)


@pytest.mark.parametrize("rel", ["b1-r1-20261012/x", "b1-r1-20261012", "b1-r2-20261012", "b1-r2-20261012/out",
                                 "l2-ab-20261012/out"])
def test_guard_out_refuses_protected_dirs_and_the_run_dir(tmp_path, rel):
    root, run_dir = _root(tmp_path)
    with pytest.raises(l2_cap1_eval.RefusedOut):
        l2_cap1_eval.guard_out(os.path.join(root, rel), run_dir=run_dir, root=root)


def test_guard_out_refuses_a_path_outside_the_root(tmp_path):
    root, run_dir = _root(tmp_path)
    with pytest.raises(l2_cap1_eval.RefusedOut):
        l2_cap1_eval.guard_out(str(tmp_path / "elsewhere"), run_dir=run_dir, root=root)
    with pytest.raises(l2_cap1_eval.RefusedOut):
        l2_cap1_eval.guard_out(root, run_dir=run_dir, root=root)


def test_guard_out_refuses_inside_an_unprotected_run_dir(tmp_path):
    root, _ = _root(tmp_path)
    run_dir = os.path.join(root, "some-run")
    os.makedirs(run_dir)
    with pytest.raises(l2_cap1_eval.RefusedOut):
        l2_cap1_eval.guard_out(run_dir, run_dir=run_dir, root=root)
    with pytest.raises(l2_cap1_eval.RefusedOut):
        l2_cap1_eval.guard_out(os.path.join(run_dir, "l2-cap1"), run_dir=run_dir, root=root)


def test_guard_out_accepts_a_fresh_sibling_under_the_root(tmp_path):
    root, run_dir = _root(tmp_path)
    out = os.path.join(root, "l2-cap1-20261013")
    assert l2_cap1_eval.guard_out(out, run_dir=run_dir, root=root) == os.path.realpath(out)


def test_guard_out_refuses_the_real_r1_run_dir():
    with pytest.raises(l2_cap1_eval.RefusedOut):
        l2_cap1_eval.guard_out("/mnt/nvme1/nvfp4-work/b1-r1-20261012/x",
                               run_dir="/mnt/nvme1/nvfp4-work/b1-r1-20261012")


def test_main_returns_2_on_a_refused_out_without_touching_the_filesystem(tmp_path):
    run_dir = tmp_path / "run"
    out = run_dir / "out"
    argv = [str(run_dir), "--dataset", str(tmp_path / "d"), "--corpus", str(tmp_path / "c"),
            "--b1-head", str(tmp_path / "h.pt"), "--out", str(out), "--cost", f"L2_T0={tmp_path / 'c0.json'}"]
    assert l2_cap1_eval.main(argv) == 2
    assert list(tmp_path.iterdir()) == []


def test_main_returns_2_on_the_r1_run_dir():
    argv = ["/mnt/nvme1/nvfp4-work/b1-r1-20261012", "--dataset", "d", "--corpus", "c", "--b1-head", "h",
            "--out", "/mnt/nvme1/nvfp4-work/b1-r1-20261012/l2-cap1"]
    assert l2_cap1_eval.main(argv) == 2
    assert not os.path.exists("/mnt/nvme1/nvfp4-work/b1-r1-20261012/l2-cap1")


@pytest.mark.parametrize("text", ["L2_T0", "=a.json", "L2_T0=", ""])
def test_cost_arg_refuses_malformed_name_path(text):
    with pytest.raises(argparse.ArgumentTypeError):
        l2_cap1_eval._cost_arg(text)


def test_cost_arg_parses_and_the_parser_exits_on_malformed():
    assert l2_cap1_eval._cost_arg("L2_T3=/x/c=1.json") == ("L2_T3", "/x/c=1.json")
    with pytest.raises(SystemExit):
        l2_cap1_eval._parser().parse_args(["r", "--dataset", "d", "--corpus", "c", "--b1-head", "h", "--out", "o",
                                           "--cost", "nonsense"])


def test_cost_paths_refuses_unknown_and_duplicate_names():
    assert l2_cap1_eval._cost_paths([("L2_T0", "a"), ("L2_T3", "b")]) == {"L2_T0": "a", "L2_T3": "b"}
    with pytest.raises(ValueError):
        l2_cap1_eval._cost_paths([("L2_T1", "a")])
    with pytest.raises(ValueError):
        l2_cap1_eval._cost_paths([("L2_T0", "a"), ("L2_T0", "b")])


def test_logged_commands_reads_the_run_script_log_or_nothing(tmp_path):
    assert l2_cap1_eval._logged_commands(str(tmp_path)) == []
    (tmp_path / "commands.txt").write_text("git_head=abc\n\nCMD: r1_cost.py --cap 1\nEXIT=0\n")
    assert l2_cap1_eval._logged_commands(str(tmp_path)) == ["git_head=abc", "CMD: r1_cost.py --cap 1", "EXIT=0"]



def _groups(n: int) -> list:
    return [SimpleNamespace(key=SimpleNamespace(session=f"s{i % 2}", block="b0", seq=i, source_layer=7,
                                                target_layer=9, horizon=2)) for i in range(n)]


def test_score_cache_round_trips_and_is_invalidated_by_sha_or_keys(tmp_path):
    path = str(tmp_path / "scores" / "dev" / "L2_T0.npz")
    keys = l2_cap1_eval.group_ids(_groups(3))
    scores = [np.arange(4, dtype=np.float32) + i for i in range(3)]
    l2_cap1_eval.save_scores(path, keys=keys, sha="aaa", scores=scores)
    got = l2_cap1_eval.load_scores(path, keys=keys, sha="aaa")
    assert got is not None and all(np.array_equal(a, b) for a, b in zip(got, scores))
    assert all(a.dtype == np.float32 for a in got)
    assert l2_cap1_eval.load_scores(path, keys=keys, sha="bbb") is None
    assert l2_cap1_eval.load_scores(path, keys=keys[:2], sha="aaa") is None
    assert l2_cap1_eval.load_scores(str(tmp_path / "missing.npz"), keys=keys, sha="aaa") is None
    assert [p.name for p in (tmp_path / "scores" / "dev").iterdir()] == ["L2_T0.npz"]


class _Counting:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def score_groups(self, groups):
        self.calls += 1
        return [np.full(4, float(self.calls), dtype=np.float32) for _ in groups]


def test_cached_scorer_reuses_matching_scores_and_rescores_on_a_sha_change(tmp_path):
    path = str(tmp_path / "scores" / "calib" / "L2_T3.npz")
    groups = _groups(2)
    inner = _Counting("L2_T3")
    first = l2_cap1_eval._CachedScorer(inner, path=path, sha="aaa").score_groups(groups)
    again = l2_cap1_eval._CachedScorer(inner, path=path, sha="aaa").score_groups(groups)
    assert inner.calls == 1 and all(np.array_equal(a, b) for a, b in zip(first, again))
    changed = l2_cap1_eval._CachedScorer(inner, path=path, sha="ccc").score_groups(groups)
    assert inner.calls == 2 and float(changed[0][0]) == 2.0
    assert l2_cap1_eval._CachedScorer(inner, path=path, sha="ccc").name == "L2_T3"
