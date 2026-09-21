"""overlap_timeline: per-row causal overlap of storage reads and CPU packing, from stage records."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap_timeline as ot  # noqa: E402


def _record(rows, extents, *, submit=1000, status="served", schema=2, **extra):
    """rows: [(start, end)] and, optionally, admit; extents: [(row, cqe)] or (row, cqe, submit)."""
    row_pack = [
        {"row": k, "start": r[0], "end": r[1], **({"admit": r[2]} if len(r) > 2 else {})} for k, r in enumerate(rows)
    ]
    extent_cqe = []
    for e in extents:
        entry = {"row": e[0], "part": 0, "cqe": e[1]}
        if len(e) > 2:
            entry.update(submit=e[2], attempts=0)
        extent_cqe.append(entry)
    return {
        "kind": "ram_miss_request",
        "schema": schema,
        "status": status,
        "layer": 3,
        "request": {"seq": 1},
        "stages_ns": {"submit": submit},
        "row_pack_ns": row_pack,
        "extent_cqe_ns": extent_cqe,
        "untraced": {"rows": 0, "extents": 0},
        **extra,
    }


def test_rows_packing_while_a_later_row_is_outstanding_overlap():
    result = ot.analyse_request(
        _record([(2000, 3000), (4000, 5000), (6000, 7000)], [(0, 2000), (1, 4000), (2, 6000)])
    )
    assert result["overlapped_rows"] == 2  # rows 0 and 1; row 2 packs after every completion
    assert result["window_ns"] == 5000 and result["pack_ns"] == 3000
    assert result["hidden_ns"] == 2000  # row 2's packing lies past the last completion
    assert result["hidden_fraction_of_pack"] == pytest.approx(2 / 3)
    assert result["saved_ns"] == 2000  # 5000 + 3000 packed serially, against 6000
    assert result["exposed_tail_ns"] == 1000
    assert result["chain_violations"] == 0


def test_a_reader_that_packs_only_after_the_last_completion_shows_no_overlap():
    result = ot.analyse_request(_record([(6000, 7000), (7000, 8000)], [(0, 2000), (1, 3000)]))
    assert result["overlapped_rows"] == 0 and result["hidden_ns"] == 0 and result["saved_ns"] == 0


def test_rows_completed_together_are_not_an_overlap_however_they_interleave_with_packing():
    """Both rows were reaped at 2000: row 0's packing runs while row 1 waits for the PACKER, not for storage."""
    result = ot.analyse_request(_record([(2000, 3000), (3000, 4000)], [(0, 2000), (1, 2000)]))
    assert result["overlapped_rows"] == 0 and result["hidden_ns"] == 0
    assert result["ready_to_pack_ns"] == [0, 1000]


def test_a_row_is_ready_when_its_last_extent_is_reaped_whatever_the_order_of_the_stamps():
    """Two extents of row 0, the later reaped listed first: the row's readiness is the later one."""
    result = ot.analyse_request(_record([(3500, 4000), (4500, 5000)], [(0, 3500), (0, 2000), (1, 4500)]))
    assert result["ready_to_pack_ns"] == [0, 0] and result["chain_violations"] == 0


def test_a_row_packed_before_its_own_extent_completed_is_a_chain_violation():
    result = ot.analyse_request(_record([(4000, 5000), (6000, 7000)], [(0, 5000), (1, 6000)]))
    assert result["chain_violations"] == 1


@pytest.mark.parametrize(
    "record, reason",
    [
        (_record([(2000, 3000)], [(0, 2000)], status="touch"), "status touch"),
        (_record([(2000, 3000)], [(0, 2000)], untraced={"rows": 1, "extents": 0}), "beyond the trace's bounds"),
        (_record([(0, 0)], [(0, 2000)]), "never packed"),
        (_record([(2000, 3000)], [(0, 0)]), "no completion stamp"),
        (_record([(2000, 3000)], [(0, 2000)], submit=0), "no submit stamp"),
    ],
)
def test_a_request_that_cannot_be_judged_is_skipped_with_its_reason(record, reason):
    assert reason in ot.analyse_request(record)["skipped"]


def test_schema_3_reports_queueing_and_flags_a_broken_admit_submit_cqe_chain():
    good = ot.analyse_request(
        _record([(2000, 3000, 900)], [(0, 2000, 1000)], schema=3)  # admit 900 <= submit 1000 <= cqe 2000 <= start
    )
    assert good["schema3_violations"] == 0 and good["queue_ns"] == [100]
    late_submit = ot.analyse_request(_record([(2000, 3000, 1500)], [(0, 2000, 1000)], schema=3))  # submit before admit
    assert late_submit["schema3_violations"] == 1
    after_cqe = ot.analyse_request(_record([(2500, 3000, 900)], [(0, 2000, 2200)], schema=3))  # submit after cqe
    assert after_cqe["schema3_violations"] == 1


def test_a_schema_1_file_is_refused_not_approximated(tmp_path):
    path = tmp_path / "old.trace"
    old = {"kind": "ram_miss_request", "stages_ns": {"submit": 1}, "request": {"seq": 1}}
    path.write_text(json.dumps(old) + "\n")
    with pytest.raises(ot.UnsupportedSchema, match="schema 1"):
        ot.analyse_file(str(path))


def test_a_file_is_summarised_over_its_multi_row_served_requests(tmp_path):
    overlapping = _record([(2000, 3000), (4000, 5000)], [(0, 2000), (1, 4000)])
    single = _record([(2000, 3000)], [(0, 2000)])
    touch = _record([], [], status="touch")
    forward_call = {"kind": "graph_step", "forward": 1}  # not a stage record
    path = tmp_path / "t.trace"
    path.write_text("\n".join(json.dumps(r) for r in (overlapping, single, touch, forward_call)) + "\n")
    result = ot.analyse_file(str(path))
    assert result["schemas"] == {2: 3} and result["multi_row_requests"] == 1 and result["single_row_requests"] == 1
    assert result["skipped"] == {"status touch": 1}
    assert result["requests_with_overlap"] == 1 and result["chain_violations_cqe_after_pack_start"] == 0
    assert result["rows_packed_while_another_outstanding"] == 1  # row 0 only
    assert result["example"] is None  # fewer rows than the drawing needs
    assert "requests with overlap: 1/1" in ot.format_report(result)


def test_the_timeline_marks_reads_and_packing_per_row():
    text = ot.render_timeline(_record([(2000, 3000), (4000, 5000)], [(0, 2000), (1, 4000)]), width=41)
    lines = text.splitlines()
    assert len(lines) == 3 and "#" in lines[1] and "." in lines[2]
    row0, row1 = lines[1].split("|")[1], lines[2].split("|")[1]
    assert row0.index("#") < row1.index("#")  # row 0 packs first
    assert row1.index(".") == 0 and row1.rstrip().endswith("#")  # row 1's read is outstanding from the submit
