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
