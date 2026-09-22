"""native_mirror_report warns about trace schemas only where the latency spans really differ."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_mirror_report as report  # noqa: E402


def _line(schema=None):
    line = {
        "kind": "ram_miss_request",
        "request": {"type": "demand", "ok": 1},
        "bytes": 10,
        "extents": 1,
        "drives": [{"dev": 1, "bytes": 10, "extents": 1}],
        "spans_ns": {"submit_to_first_cqe": 1000, "first_to_last_cqe": 2000, "pack": 3000},
    }
    if schema is not None:  # schema 1 lines have no schema field
        line["schema"] = schema
    return line


def _trace(tmp_path, *schemas):
    path = tmp_path / "t.trace"
    path.write_text("\n".join(json.dumps(_line(s)) for s in schemas) + "\n")
    return str(path)


def _summarize(tmp_path, capsys, *schemas):
    result = report.summarize("arm", _trace(tmp_path, *schemas), {1: "nvme0"})
    return result, capsys.readouterr().out


@pytest.mark.parametrize("mix", [(2, 3), (3, 4), (2, 3, 4)], ids=["2+3", "3+4", "2+3+4"])
def test_a_mix_of_schemas_with_the_same_span_meaning_does_not_warn(tmp_path, capsys, mix):
    _, out = _summarize(tmp_path, capsys, *mix)
    assert "WARNING" not in out
    assert "differ only in added fields" in out


@pytest.mark.parametrize("mix", [(None, 2), (None, 3), (None, 4)], ids=["1+2", "1+3", "1+4"])
def test_a_mix_across_the_first_batch_boundary_warns(tmp_path, capsys, mix):
    _, out = _summarize(tmp_path, capsys, *mix)
    assert "WARNING mixed trace schemas [1," in out


def test_a_single_schema_says_what_its_spans_measure(tmp_path, capsys):
    _, out = _summarize(tmp_path, capsys, 4, 4)
    assert "WARNING" not in out and "trace schema 4: spans cover the whole read" in out
    _, out = _summarize(tmp_path, capsys, None, None)
    assert "WARNING" not in out and "trace schema 1: spans are the first batch's" in out


@pytest.mark.parametrize(
    "base, mirror, warns",
    [
        ({3}, {4}, False),
        ({2}, {3, 4}, False),
        ({1}, {2}, True),
        ({1}, {4}, True),
        ({4}, {4}, False),
    ],
)
def test_two_arms_are_compared_only_when_their_spans_mean_the_same(base, mirror, warns):
    assert (report.span_kinds(base) != report.span_kinds(mirror)) is warns


def _run_main(tmp_path, monkeypatch, capsys, base_schema, mirror_schema):
    sessions = tmp_path / "s.json"
    sessions.write_text(json.dumps({"per_session": [{"decode_tok_s": 1.0}]}))
    base = tmp_path / "base.trace"
    mirror = tmp_path / "mirror.trace"
    base.write_text(json.dumps(_line(base_schema)) + "\n")
    mirror.write_text(json.dumps(_line(mirror_schema)) + "\n")
    monkeypatch.setattr(sys, "argv", ["report", str(sessions), str(base), str(sessions), str(mirror)])
    report.main()
    return capsys.readouterr().out


@pytest.mark.parametrize("base, mirror", [(3, 4), (2, 4)], ids=["3-vs-4", "2-vs-4"])
def test_the_arm_comparison_stays_quiet_when_spans_mean_the_same(tmp_path, monkeypatch, capsys, base, mirror):
    out = _run_main(tmp_path, monkeypatch, capsys, base, mirror)
    assert "NOT comparable" not in out and "BYTE PARITY" in out


@pytest.mark.parametrize("base, mirror", [(None, 4), (4, None)], ids=["1-vs-4", "4-vs-1"])
def test_the_arm_comparison_warns_across_the_first_batch_boundary(tmp_path, monkeypatch, capsys, base, mirror):
    out = _run_main(tmp_path, monkeypatch, capsys, base, mirror)
    assert "the latency spans above are NOT comparable" in out
