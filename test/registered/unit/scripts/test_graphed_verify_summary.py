"""The D2-3 driver's summary of an eager, a graphed and a re-verify-all arm (CPU)."""

import json
import os
import sys

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "dspark")
)
import graphed_verify  # noqa: E402


def _arm(outdir, arm, texts, verify_ms, overflow=None, graphed=None, truncated=None):
    sessions = [
        {"decode_tok_s": 10.0 + i, "completion_tokens": 30, "spec_verify_ct": 10, "output_text": text}
        for i, text in enumerate(texts)
    ]
    report = {
        "per_session": sessions,
        "mean_decode_tok_s": 10.5,
        "dspark_info_record": {"records": [{"target_verify_gpu_ms": ms} for ms in verify_ms]},
    }
    with open(os.path.join(outdir, f"{arm}.json"), "w") as f:
        json.dump(report, f)
    if overflow is not None:
        record = {"counters": {"residency_gpu": {"gather_overflow": overflow, "insertion_truncated": truncated},
                               "graphed_verify": graphed}}
        with open(os.path.join(outdir, f"{arm}.metrics.jsonl"), "w") as f:
            f.write(json.dumps({"counters": {}}) + "\n" + json.dumps(record) + "\n")


@pytest.mark.parametrize("arm", sorted(graphed_verify.D23_ARMS))
def test_no_arm_names_a_core_by_hand(tmp_path, arm):
    """ThreadingConfig derives the RAM thread's and the draft's cores from the arm's affinity. The recipe's spin core 17
    and ab_cpu_draft's draft cores 6-17 were chosen for other affinities, and the graphed arm was refused for them."""
    env = graphed_verify.arm_environment(arm, str(tmp_path))
    assert "SGLANG_DSV41_RAM_MISS_SPIN_CORE" not in env
    assert "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES" not in env
    assert env["SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS"] == "1"


def test_the_summary_reads_every_arm(tmp_path):
    out = str(tmp_path)
    _arm(out, "eager", ["a", "b"], [20.0, 30.0, None])
    _arm(out, "graphed", ["a", "c"], [10.0, 14.0], overflow=[2, 0, 5], graphed={"graphed_verify_ct": 10, "verify_overflow_ct": 6},
         truncated=[0, 0, 0])
    _arm(out, "reverify", ["a", "b"], [40.0], overflow=[0, 0, 0], graphed={"graphed_verify_ct": 10, "verify_overflow_ct": 0},
         truncated=[0, 0, 0])
    summary = graphed_verify.summarize(out)
    assert summary["eager"]["accept_length"] == 3.0 and summary["eager"]["verify_ms"]["mean"] == 25.0
    graphed = summary["graphed"]
    assert graphed["reverify_rate"] == pytest.approx(0.6)
    assert graphed["layer_overflow_rate"] == {"mean": pytest.approx(0.7 / 3), "max": pytest.approx(0.5)}
    assert graphed["insertion_truncated"] == 0 and graphed["text_matches_eager"] == 1
    assert summary["reverify"]["text_matches_eager"] == 2
    with open(os.path.join(out, "summary.json")) as f:
        assert json.load(f) == summary


def test_the_draft_arms_summary(tmp_path):
    out = str(tmp_path)
    for arm, texts in (("draft-eager", ["a", "b"]), ("draft-graph", ["a", "c"])):
        _arm(out, arm, texts, [10.0], overflow=[0], graphed={"graphed_verify_ct": 4, "verify_overflow_ct": 4}, truncated=[0])
        with open(os.path.join(out, f"{arm}.json")) as f:
            report = json.load(f)
        report["dspark_info_record"]["records"] = [{"target_verify_gpu_ms": 10.0, "draft_gpu_ms": 30.0 if arm == "draft-eager" else 12.0}]
        with open(os.path.join(out, f"{arm}.json"), "w") as f:
            json.dump(report, f)
    with open(os.path.join(out, "draft-graph.log"), "w") as f:
        f.write("DSpark: EXL3 draft graphs on (3 draft MoE layers prepared)\n")
        f.write("DSpark CPU experts: {'jobs': 5, 'rows': 40, 'forward_ns': 2000000, 'keep_warm_calls': 7}\n")
    summary = graphed_verify.summarize(out)
    assert summary["draft-graph"]["text_matches_draft_eager"] == [True, False]
    assert summary["draft-graph"]["draft_cpu"] == {"jobs": 5, "rows": 40, "forward_ns": 2000000, "keep_warm_calls": 7}
    assert summary["draft-graph"]["draft_graphs_on"] and not summary["draft-eager"]["draft_graphs_on"]
    assert summary["draft-eager"]["draft_cpu"] is None
    assert summary["draft-graph"]["draft_gpu_ms"]["mean"] == 12.0


def test_the_draft_eager_arm_only_differs_by_the_switch(tmp_path):
    eager = graphed_verify.arm_environment("draft-eager", str(tmp_path))
    graph = graphed_verify.arm_environment("draft-graph", str(tmp_path))
    assert eager.pop("SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH") == "1"
    assert "SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH" not in graph
    assert eager.pop("SGLANG_MOE_HOT_METRICS_FILE") != graph.pop("SGLANG_MOE_HOT_METRICS_FILE")
    assert eager == graph
