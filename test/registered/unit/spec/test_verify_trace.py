import json
import os
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import verify_trace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GREEDY = {"all_greedy": True, "penalties": False, "logit_bias": False, "grammar": False}


def _row(rid, index, draft, argmax, accept_len, gap=None, committed=None, **flags):
    return {
        "rid": rid,
        "verify_index": index,
        "draft": draft,
        "argmax": argmax,
        "accept_len": accept_len,
        "committed": argmax[:accept_len] if committed is None else committed,
        "gap": gap or [0.5, 0.5, 0.5, 0.5],
        **GREEDY,
        **flags,
    }


def _sampling_info(**changes):
    fields = dict(
        is_all_greedy=True,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
        logit_bias=None,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_trace_is_off_unless_the_env_names_a_file():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_SPEC_VERIFY_TRACE", None)
        verify_trace.verify_trace_path.cache_clear()
        assert verify_trace.verify_trace_path() == ""
    with mock.patch.dict(os.environ, {"SGLANG_SPEC_VERIFY_TRACE": "/tmp/trace.jsonl"}):
        verify_trace.verify_trace_path.cache_clear()
        assert verify_trace.verify_trace_path() == "/tmp/trace.jsonl"
    verify_trace.verify_trace_path.cache_clear()


def test_trace_path_carries_the_rank_when_several_ranks_run():
    with (
        mock.patch.dict(os.environ, {"SGLANG_SPEC_VERIFY_TRACE": "/tmp/trace.jsonl"}),
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch.object(torch.distributed, "get_world_size", return_value=2),
        mock.patch.object(torch.distributed, "get_rank", return_value=1),
    ):
        verify_trace.verify_trace_path.cache_clear()
        assert verify_trace.verify_trace_path() == "/tmp/trace.jsonl.rank1"
    verify_trace.verify_trace_path.cache_clear()


def test_logit_summary_gives_argmax_and_top_two_gap_per_verify_row():
    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0],
            [2.0, 0.0, 1.5],
            [0.0, 0.0, 4.0],
            [1.0, 1.25, 0.0],
            [5.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.75],
        ]
    )

    argmax, gaps = verify_trace.verify_logit_summary(logits, draft_token_num=4)

    assert argmax.tolist() == [[1, 0, 2, 1], [0, 1, 0, 2]]
    assert torch.allclose(
        gaps, torch.tensor([[2.0, 0.5, 4.0, 0.25], [4.0, 1.0, 1.0, 0.25]])
    )


def test_sampling_flags_mark_every_transform_that_moves_committed_tokens():
    assert verify_trace.sampling_flags(_sampling_info(), False) == GREEDY
    flagged = verify_trace.sampling_flags(
        _sampling_info(
            is_all_greedy=False,
            penalizer_orchestrator=SimpleNamespace(is_required=True),
            logit_bias=torch.zeros(1, 3),
        ),
        True,
    )
    assert flagged == {
        "all_greedy": False,
        "penalties": True,
        "logit_bias": True,
        "grammar": True,
    }
    assert verify_trace.sampling_flags(
        _sampling_info(acc_additive_penalties=torch.zeros(1, 3)), False
    )["penalties"]


def test_record_numbers_verifies_per_request_and_writes_jsonl(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    verify_trace._writer.cache_clear()
    logits = torch.zeros(4, 3)
    logits[:, 2] = 1.0
    summary = verify_trace.verify_logit_summary(logits, draft_token_num=4)

    for rid in ("a", "b", "a"):
        verify_trace.record_verify_trace(
            path,
            [SimpleNamespace(rid=rid)],
            torch.tensor([7, 8, 9, 10]),
            summary,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([2, 2, 5, 2], dtype=torch.int32),
            torch.tensor([[0, 1, 2, -1]], dtype=torch.int32),
            GREEDY,
        )

    rows = [json.loads(line) for line in open(path)]
    assert [(row["rid"], row["verify_index"]) for row in rows] == [
        ("a", 0),
        ("b", 0),
        ("a", 1),
    ]
    assert rows[0]["draft"] == [7, 8, 9, 10]
    assert rows[0]["argmax"] == [2, 2, 2, 2]
    assert rows[0]["accept_len"] == 3
    assert rows[0]["committed"] == [2, 2, 5]
    assert rows[0]["gap"] == [1.0, 1.0, 1.0, 1.0]
    assert {key: rows[0][key] for key in GREEDY} == GREEDY
    verify_trace._writer.cache_clear()


def test_record_splits_a_two_request_batch_per_request(tmp_path):
    path = str(tmp_path / "trace.jsonl")
    verify_trace._writer.cache_clear()
    logits = torch.zeros(8, 40)
    for row, token in enumerate([11, 12, 13, 14, 21, 22, 23, 24]):
        logits[row, token] = 1.0 + row
    summary = verify_trace.verify_logit_summary(logits, draft_token_num=4)

    verify_trace.record_verify_trace(
        path,
        [SimpleNamespace(rid="first"), SimpleNamespace(rid="second")],
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        summary,
        torch.tensor([2, 4], dtype=torch.int32),
        torch.tensor([31, 32, 33, 34, 35, 36, 37, 38], dtype=torch.int32),
        torch.tensor([[0, 1, -1, -1], [4, 5, 6, 7]], dtype=torch.int32),
        GREEDY,
    )

    first, second = [json.loads(line) for line in open(path)]
    assert (first["rid"], first["draft"], first["argmax"]) == (
        "first",
        [1, 2, 3, 4],
        [11, 12, 13, 14],
    )
    assert (first["accept_len"], first["committed"]) == (2, [31, 32])
    assert first["gap"] == [1.0, 2.0, 3.0, 4.0]
    assert (second["rid"], second["draft"], second["argmax"]) == (
        "second",
        [5, 6, 7, 8],
        [21, 22, 23, 24],
    )
    assert (second["accept_len"], second["committed"]) == (4, [35, 36, 37, 38])
    assert second["gap"] == [5.0, 6.0, 7.0, 8.0]
    verify_trace._writer.cache_clear()


def test_identical_traces_have_no_divergence_even_with_different_rids():
    reference = [_row("r1", 0, [1, 2, 3, 4], [1, 2, 9, 9], 3), _row("r2", 0, [5, 6, 7, 8], [5, 6, 7, 8], 4)]
    candidate = [_row("x", 0, [1, 2, 3, 4], [1, 2, 9, 9], 3), _row("y", 0, [5, 6, 7, 8], [5, 6, 7, 8], 4)]

    report = verify_trace.compare_traces(reference, candidate)

    assert [entry["request"] for entry in report] == [0, 1]
    assert all(entry["comparable"] for entry in report)
    assert all(entry["first_any_divergence"] is None for entry in report)
    assert all(entry["first_committed_divergence"] is None for entry in report)


def test_argmax_noise_past_the_commit_is_not_a_committed_divergence():
    report = verify_trace.compare_request(
        [_row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 4), _row("r", 1, [5, 6, 7, 8], [5, 0, 7, 8], 1, [2.0, 0.05, 1.0, 1.0])],
        [_row("c", 0, [1, 2, 3, 4], [1, 2, 3, 4], 4), _row("c", 1, [5, 6, 7, 8], [5, 9, 7, 8], 1, [2.0, 0.04, 1.0, 1.0])],
    )
    assert report["first_any_divergence"] == {
        "verify_index": 1,
        "field": "argmax",
        "row": 1,
        "reference_gap": 0.05,
        "candidate_gap": 0.04,
    }
    assert report["first_committed_divergence"] is None


def test_equal_accept_lengths_diverge_only_inside_the_commit():
    inside = verify_trace.compare_request(
        [_row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 3, [1.0, 1.0, 3.5, 1.0])],
        [_row("c", 0, [1, 2, 3, 4], [1, 2, 8, 4], 3, [1.0, 1.0, 3.0, 1.0])],
    )
    assert inside["first_committed_divergence"] == {
        "verify_index": 0,
        "field": "committed",
        "row": 2,
        "reference_gap": 3.5,
        "candidate_gap": 3.0,
    }

    past = verify_trace.compare_request(
        [_row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 3)],
        [_row("c", 0, [1, 2, 3, 4], [1, 2, 3, 8], 3)],
    )
    assert past["first_any_divergence"]["row"] == 3
    assert past["first_committed_divergence"] is None


def test_a_later_committed_divergence_is_reported_after_an_uncommitted_one():
    report = verify_trace.compare_request(
        [
            _row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 2),
            _row("r", 1, [5, 6, 7, 8], [5, 6, 7, 8], 4, [1.0, 1.0, 1.0, 0.2]),
        ],
        [
            _row("c", 0, [1, 2, 3, 4], [1, 2, 7, 4], 2),
            _row("c", 1, [5, 6, 7, 8], [5, 6, 7, 9], 4, [1.0, 1.0, 1.0, 0.1]),
        ],
    )
    assert report["first_any_divergence"]["verify_index"] == 0
    assert report["first_any_divergence"]["field"] == "argmax"
    assert report["first_committed_divergence"] == {
        "verify_index": 1,
        "field": "committed",
        "row": 3,
        "reference_gap": 0.2,
        "candidate_gap": 0.1,
    }


def test_committed_tokens_not_argmax_decide_the_committed_divergence():
    report = verify_trace.compare_request(
        [_row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 2, committed=[1, 5])],
        [_row("c", 0, [1, 2, 3, 4], [1, 2, 3, 4], 2, committed=[1, 6])],
    )
    assert report["first_any_divergence"]["field"] == "committed"
    assert report["first_committed_divergence"]["field"] == "committed"
    assert report["first_committed_divergence"]["row"] == 1


def test_draft_and_accept_length_divergences():
    report = verify_trace.compare_request(
        [_row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 4)],
        [_row("c", 0, [1, 7, 3, 4], [1, 2, 3, 4], 2)],
    )
    assert report["first_any_divergence"]["field"] == "draft"
    assert report["first_any_divergence"]["row"] == 1
    assert report["first_committed_divergence"]["field"] == "accept_len"
    assert report["first_committed_divergence"]["row"] == 2


def test_a_flagged_or_non_greedy_row_makes_the_request_not_comparable():
    clean = [_row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 4)]
    for flags in ({"all_greedy": False}, {"penalties": True}, {"logit_bias": True}, {"grammar": True}):
        flagged = [clean[0], _row("c", 1, [1, 2, 3, 4], [1, 2, 3, 4], 4, **flags)]
        assert verify_trace.compare_request(clean + clean, flagged)["comparable"] is False
        assert verify_trace.compare_request(flagged, clean + clean)["comparable"] is False
    assert verify_trace.compare_request(clean, clean)["comparable"] is True


def test_compare_reports_unequal_verify_counts():
    report = verify_trace.compare_traces(
        [_row("r", 0, [1, 2, 3, 4], [1, 2, 3, 4], 4), _row("r", 1, [1, 2, 3, 4], [1, 2, 3, 4], 4)],
        [_row("c", 0, [1, 2, 3, 4], [1, 2, 3, 4], 4)],
    )
    assert report[0]["reference_verifies"] == 2
    assert report[0]["candidate_verifies"] == 1
    for key in ("first_any_divergence", "first_committed_divergence"):
        assert report[0][key]["field"] == "verify_count"
        assert report[0][key]["verify_index"] == 1
