"""CPU-only tests for the parity comparison: a gate that cannot fail, or compares unlike dumps, proves nothing."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import compare
import dump_format

K = 20
N = 64
MANIFEST = {"prompt_set_sha256": "abc", "git_commit": "deadbeef", "sglang_file": "/x/python/sglang/__init__.py"}


def _softmax_top(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    order = np.argsort(-logp, axis=-1)[:, :K]
    return order.astype(np.int32), np.take_along_axis(logp, order, axis=-1).astype(np.float32)


def _prompt(logits: np.ndarray, tokens: np.ndarray) -> dict:
    top_ids, top_lp = _softmax_top(logits)
    logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    return {
        "tokens": tokens,
        "chosen_lp": logp[np.arange(len(tokens)), tokens].astype(np.float32),
        "top_ids": top_ids,
        "top_lp": top_lp,
    }


def _logits(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(N, 500)) * 2.0


def _write(path: Path, logits_by_prompt: dict, *, manifest: dict | None = None, tokens_by_prompt: dict | None = None):
    prompts = {}
    for prompt_id, logits in logits_by_prompt.items():
        tokens = (tokens_by_prompt or {}).get(prompt_id, np.arange(N, dtype=np.int32))
        prompts[prompt_id] = _prompt(logits, tokens)
    dump_format.write_dump(out_dir=path, manifest=manifest or MANIFEST, prompts=prompts)


def test_identical_dumps_pass_with_zero_kl(tmp_path):
    logits = {"a": _logits(1), "b": _logits(2)}
    _write(tmp_path / "base", logits)
    _write(tmp_path / "branch", logits)
    result = compare.compare(base_dir=tmp_path / "base", branch_dir=tmp_path / "branch")
    assert result["mean_kl"] < 1e-9 and result["top1_agreement"] == 1.0
    assert compare.main([str(tmp_path / "base"), str(tmp_path / "branch")]) == compare.EXIT_PASS


def test_small_noise_stays_under_the_default_threshold(tmp_path):
    base = {"a": _logits(1)}
    noise = np.random.default_rng(9).normal(size=(N, 500)) * 0.005
    _write(tmp_path / "base", base)
    _write(tmp_path / "branch", {"a": base["a"] + noise})
    assert compare.main([str(tmp_path / "base"), str(tmp_path / "branch")]) == compare.EXIT_PASS


def test_shifted_distribution_fails_both_gates(tmp_path):
    base = {"a": _logits(1)}
    _write(tmp_path / "base", base)
    _write(tmp_path / "branch", {"a": base["a"] + np.random.default_rng(3).normal(size=(N, 500)) * 1.5})
    result = compare.compare(base_dir=tmp_path / "base", branch_dir=tmp_path / "branch")
    assert result["mean_kl"] > 1e-3 and result["top1_agreement"] < 0.995
    assert compare.main([str(tmp_path / "base"), str(tmp_path / "branch")]) == compare.EXIT_FAIL


def test_thresholds_are_cli_overridable(tmp_path):
    base = {"a": _logits(1)}
    _write(tmp_path / "base", base)
    _write(tmp_path / "branch", {"a": base["a"] + np.random.default_rng(3).normal(size=(N, 500)) * 1.5})
    args = [str(tmp_path / "base"), str(tmp_path / "branch"), "--max-kl", "100", "--min-top1", "0"]
    assert compare.main(args) == compare.EXIT_PASS


def test_unequal_top_k_support_counts_the_missing_mass():
    # Branch moves all probability from id 0 to id 1; both sides see only those two ids.
    base_ids = np.array([0, 1] + [-1] * (K - 2), dtype=np.int32)
    pad = [-np.inf] * (K - 2)
    base_lp = np.array([np.log(0.9), np.log(0.1)] + pad, dtype=np.float32)
    branch_lp = np.array([np.log(0.1), np.log(0.9)] + pad, dtype=np.float32)
    kl = compare.position_kl(base_ids=base_ids, base_lp=base_lp, branch_ids=base_ids, branch_lp=branch_lp)
    expected = 0.9 * np.log(0.9 / 0.1) + 0.1 * np.log(0.1 / 0.9)
    assert abs(kl - expected) < 1e-6


def test_mismatched_prompt_sets_are_refused(tmp_path):
    logits = {"a": _logits(1)}
    _write(tmp_path / "base", logits)
    _write(tmp_path / "branch", logits, manifest={**MANIFEST, "prompt_set_sha256": "other"})
    assert compare.main([str(tmp_path / "base"), str(tmp_path / "branch")]) == compare.EXIT_REFUSED


def test_mismatched_prompt_ids_are_refused(tmp_path):
    _write(tmp_path / "base", {"a": _logits(1)})
    _write(tmp_path / "branch", {"b": _logits(1)})
    assert compare.main([str(tmp_path / "base"), str(tmp_path / "branch")]) == compare.EXIT_REFUSED


def test_mismatched_token_ids_are_refused(tmp_path):
    logits = {"a": _logits(1)}
    _write(tmp_path / "base", logits)
    _write(tmp_path / "branch", logits, tokens_by_prompt={"a": np.arange(N, dtype=np.int32) + 1})
    assert compare.main([str(tmp_path / "base"), str(tmp_path / "branch")]) == compare.EXIT_REFUSED
