"""CPU tests for l2_fast_report: pending rendering, labels on every number, near/far disagreement separation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l2_fast_report as rep  # noqa: E402

_STATS = {"median_ms": 0.05, "p90_ms": 0.06, "p99_ms": 0.07, "runs": 1000, "nodes": 3}
_MODELS = ("L2_T0", "L2_T3")
_SHA = {"L2_T0": "e6f7661d" + "0" * 56, "L2_T3": "e675e2bd" + "1" * 56}
_COMMANDS = ("git_head=abc123\nsglang.__file__=/wt/python/sglang/__init__.py\n"
             "CMD: l2_fast_cost.py --out X\nEXIT=0\n")


def _profile() -> dict:
    comp = {c: dict(_STATS) for c in rep.COMPONENTS}
    return {"label": rep.OBSERVED, "scope": "six tokens, CUDA graph", "tokens": 6, "runs": 1000, "layer": 12,
            "models": {m: {"checkpoint_sha256": _SHA[m], "components": comp} for m in _MODELS}}


def _cost() -> dict:
    per = {str(w): dict(_STATS) for w in (1, 2, 4, 6)}
    models = {m: {"checkpoint_sha256": _SHA[m], "native": per, "fused": per, "torch": per,
                  "incremental_ms": {str(w): 0.02 * (1 + i) for w in (1, 2, 4, 6)}}
              for i, m in enumerate(_MODELS)}
    return {"label": rep.OBSERVED, "scope": "widths 1,2,4,6; 38 slots", "widths": [1, 2, 4, 6], "slots": 38,
            "runs": 1000, "bar_ms_per_layer": 0.1, "models": models,
            "native_bandwidth_gbps": {str(w): 100.0 for w in (1, 2, 4, 6)}}


def _interference() -> dict:
    models = {m: {"checkpoint_sha256": _SHA[m], "fused_side": dict(_STATS), "slowdown_vs_native": 1.01,
                  "slowdown_vs_none": 1.05} for m in _MODELS}
    return {"label": rep.OBSERVED, "scope": "side stream", "width": 6, "runs": 1000, "main_alone": dict(_STATS),
            "native_side": dict(_STATS), "main_load": {"label": rep.HYPOTHESIS, "copies": 2}, "models": models}


def _cell(*, near: list, far: int, recall: bool) -> dict:
    cell = {"groups": 10, "max_abs_diff": 1e-6, "p99_abs_diff": 1e-7, "top1_agree": 1.0, "read_agree": 0.8,
            "disagreements": len(near) + far, "near_threshold": near, "far_disagreements": far}
    if recall:
        cell["recall_change"] = {"ref_recall": 0.5, "got_recall": 0.49, "diff": -0.01, "eligible": 100}
    return cell


def _parity() -> dict:
    near = [{"group": 3, "session": "S-NEAR", "ref_read": "e7", "got_read": "none", "ref_score": 0.5004,
             "got_score": 0.4996}]
    mismatch = {"kernel_record_mismatch": 4}
    cases = {
        "fp32_gate": {"calib": {"L2_T0": {"kernel": {"0.75": _cell(near=[], far=0, recall=False)}}}},
        "spec_score": {"calib": {m: {"kernel": {"0.75": {**_cell(near=near, far=7, recall=True), **mismatch}}}
                                 for m in _MODELS}},
    }
    return {"label": rep.OBSERVED, "scope": {"limit": None, "devices": "cuda"},
            "gate_dtype": "bfloat16", "gate_dtype_label": "production default",
            "thresholds": {m: {"0.75": 0.5, "1.0": 0.4} for m in _MODELS},
            "ckpts": {m: {"path": f"/ck/{m}", "sha256": _SHA[m]} for m in _MODELS}, "cases": cases}


def _build(**kw) -> dict:
    args = {"commands": None, "profile": None, "cost": None, "interference": None, "parity": None}
    args.update(kw)
    return rep.build_result(out="/mnt/nvme1/nvfp4-work/x", **args)


def _numeric_leaves(node, parent_label=None, path=""):
    """(path, value, label) of every number, label being the one in its own dict."""
    if isinstance(node, dict):
        label = node.get("label")
        for k, v in node.items():
            yield from _numeric_leaves(v, label, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _numeric_leaves(v, parent_label, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, node, parent_label


def test_all_inputs_none_leave_no_measured_number_and_render_pending():
    """A missing artifact must never render as a zero or a stale number."""
    result = _build()
    nums = [p for p, _, _ in _numeric_leaves(result)]
    assert all(p.endswith("bar_ms_per_layer") for p in nums), nums
    text = rep.render_report(result)
    assert text.count(rep.PENDING) > 40
    assert "| 0 |" not in text and "| 0.0 |" not in text


def test_every_numeric_leaf_sits_beside_a_label():
    """A number without a label cannot be told apart from an unmeasured one in result.json."""
    result = _build(commands=_COMMANDS, profile=_profile(), cost=_cost(), interference=_interference(),
                    parity=_parity())
    leaves = list(_numeric_leaves(json.loads(json.dumps(result))))
    assert leaves
    bad = [(p, lab) for p, _, lab in leaves if lab not in rep.LABELS]
    assert not bad, bad[:5]
    assert result["provenance"]["git_head"] == "abc123"
    assert result["provenance"]["checkpoints"]["L2_T0"]["prefix_matches"] is True
    assert result["cost"]["incremental_vs_bar"]["L2_T0"]["6"]["within_bar"] is True
    assert result["cost"]["incremental_vs_bar"]["L2_T3"]["6"]["within_bar"] is True


def test_report_lists_near_threshold_apart_from_far_disagreements():
    """Near-threshold read changes and far ones must not be pooled into one count."""
    text = rep.render_report(_build(parity=_parity()))
    near = text.index("### Near-threshold disagreements")
    far = text.index("### Far disagreements")
    cand = text.index("### Precision candidates")
    assert near < text.index("S-NEAR") < far < cand
    assert "| 7 |" in text[far:cand]
    assert "S-NEAR" not in text[far:]


def test_report_states_gate_dtype_record_mismatch_and_fp32_section():
    """The spec_score gate dtype, the kernel record mismatch and the float32 follow-up must reach REPORT.md."""
    fp32 = {**_parity(), "gate_dtype": "float32", "gate_dtype_label": "router_fp32"}
    result = _build(parity=_parity(), parity_router_fp32=fp32)
    text = rep.render_report(result)
    assert "production default" in text and "router_fp32" in text
    assert "kernel record mismatch" in text and "| 4 |" in text
    assert result["parity_router_fp32"]["status"] == "produced"
    assert _build()["parity_router_fp32"]["status"] == rep.PENDING


def test_write_outputs_is_atomic_and_complete(tmp_path):
    """Both files land in OUT with no tmp leftovers."""
    result = _build(parity=_parity())
    result_path, report_path = rep.write_outputs(str(tmp_path / "o"), result)
    assert os.path.basename(result_path) == "result.json" and os.path.basename(report_path) == "REPORT.md"
    assert json.loads(Path(result_path).read_text())["parity"]["status"] == "produced"
    assert sorted(os.listdir(tmp_path / "o")) == ["REPORT.md", "result.json"]
