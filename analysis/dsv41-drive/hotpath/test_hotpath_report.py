"""Unit tests for Task 18's report script (plan 2026-09-29-hotpath-zero-overhead), against recorded JSON fixtures.
No server runs. Covers the final-rereview (B) findings:
- Important: shim()'s zero rule must assert the *service* thread against its measured lifecycle floor
  (malloc=mutex=cond=0, free<=1), and only report the copy thread's counts, never assert them zero.
- Minor: identity_checked() must not pass vacuously on dropped error rows, zero compared turns, or empty content.

PYTHONPATH: this directory, plus analysis/dsv41-drive/mirror3 and analysis/dsv41-drive/iopoll-cuts (hotpath_report's
own imports).
"""

import json

import pytest

import hotpath_report as hr

SERVICE_FLOOR = {"malloc": 0, "free": 1, "mutex": 0, "cond": 0}
COPY_FLOOR = {"malloc": 72, "free": 72, "mutex": 1, "cond": 1}  # measured floor: start handshake + backend init


def _dump(service=None, copy=None, threads=None):
    service = {**SERVICE_FLOOR, **(service or {})}
    copy = {**COPY_FLOOR, **(copy or {})}
    threads = threads if threads is not None else {"service": 1, "copy": 1}
    return {"threads": threads, "service": service, "copy": copy}


def _write(tmp_path, dump):
    (tmp_path / "C-shim.json").write_text(json.dumps(dump))
    return tmp_path


# --- shim(): the Important finding -------------------------------------------------------------------------------

def test_shim_passes_at_the_measured_lifecycle_floor(tmp_path):
    # service: malloc=mutex=cond=0, free=1 (std::thread teardown); copy: the CUDA backend's real, nonzero traffic.
    _write(tmp_path, _dump(copy={"mutex": 90_035_468, "malloc": 72, "free": 72, "cond": 1}))
    out = hr.shim(tmp_path)
    assert out["ok"] is True
    assert out["why"] is None


def test_shim_fails_when_service_malloc_is_above_the_floor(tmp_path):
    # This is the real bug the original run caught (results.md SS8c/SS8e): one unattributed service malloc.
    _write(tmp_path, _dump(service={"malloc": 1}))
    out = hr.shim(tmp_path)
    assert out["ok"] is False
    assert "service" in out["why"]
    assert out["counts"]["service"]["malloc"] == 1


def test_shim_fails_when_service_mutex_or_cond_is_nonzero(tmp_path):
    _write(tmp_path, _dump(service={"mutex": 1}))
    assert hr.shim(tmp_path)["ok"] is False
    _write(tmp_path, _dump(service={"cond": 1}))
    assert hr.shim(tmp_path)["ok"] is False


def test_shim_fails_when_service_free_exceeds_the_documented_floor(tmp_path):
    _write(tmp_path, _dump(service={"free": 2}))
    assert hr.shim(tmp_path)["ok"] is False


def test_shim_does_not_assert_the_copy_thread_at_all(tmp_path):
    # A large, real copy-thread count (libcuda's cuEventQuery mutex) must not fail the gate by itself.
    _write(tmp_path, _dump(copy={"malloc": 999, "free": 999, "mutex": 123_456_789, "cond": 999}))
    out = hr.shim(tmp_path)
    assert out["ok"] is True
    assert out["counts"]["copy"]["mutex"] == 123_456_789  # reported


def test_shim_fails_when_a_tracked_thread_was_never_seen(tmp_path):
    _write(tmp_path, _dump(threads={"service": 0, "copy": 1}))
    out = hr.shim(tmp_path)
    assert out["ok"] is False
    assert "never recognized" in out["why"]


def test_shim_fails_on_anything_but_exactly_one_dump(tmp_path):
    out = hr.shim(tmp_path)  # no C-shim.json at all
    assert out["ok"] is False
    assert "found 0" in out["why"]
    _write(tmp_path, _dump())
    (tmp_path / "C-shim.json.12345").write_text(json.dumps(_dump()))
    out = hr.shim(tmp_path)
    assert out["ok"] is False
    assert "found 2" in out["why"]


def test_shim_reproduces_the_recorded_task_18_run(tmp_path):
    # The actual C-shim.json numbers from the first Task 18 run (results.md SS8c / task-18-report.md): service had
    # one unattributed malloc (fixed later, 7bebddbead), copy had the real CUDA-backend floor-plus-traffic.
    _write(tmp_path, _dump(service={"malloc": 1}, copy={"mutex": 90_035_468}))
    out = hr.shim(tmp_path)
    assert out["ok"] is False
    assert out["counts"]["service"] == {"malloc": 1, "free": 1, "mutex": 0, "cond": 0}
    # the old, unfixable-by-construction assertion would also have failed on copy's mutex count; the new one must not
    assert "copy" not in out["why"]


# --- identity_checked(): the Minor finding ------------------------------------------------------------------------

def _write_results(d, rows):
    d.mkdir(exist_ok=True)
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_identity_checked_passes_on_normal_matching_turns(tmp_path):
    ref = _write_results(tmp_path / "ref", [dict(session_id="s1", turn=0, reasoning="r", content="hello")])
    new = _write_results(tmp_path / "new", [dict(session_id="s1", turn=0, reasoning="r", content="hello")])
    assert hr.identity_checked(ref, new) == {"s1/t0": True}


def test_identity_checked_rejects_a_comparison_with_zero_turns(tmp_path):
    ref = _write_results(tmp_path / "ref", [dict(session_id="s1", turn=0, error="boom")])
    new = _write_results(tmp_path / "new", [dict(session_id="s1", turn=0, error="boom")])
    with pytest.raises(ValueError, match="zero turns"):
        hr.identity_checked(ref, new)


def test_identity_checked_rejects_a_comparison_that_silently_dropped_an_error_row(tmp_path):
    # Both arms errored on s2/t0: _turns() drops it from both, so raw m.identity() would call this "identical"
    # (the one turn that remains matches). identity_checked must catch the dropped row instead.
    rows = [dict(session_id="s1", turn=0, reasoning="r", content="c"), dict(session_id="s2", turn=0, error="boom")]
    ref = _write_results(tmp_path / "ref", rows)
    new = _write_results(tmp_path / "new", rows)
    import mirror3_report as m
    assert m.identity(ref, new) == {"s1/t0": True}  # the vacuous pass this hardening exists to catch
    with pytest.raises(ValueError, match="dropped"):
        hr.identity_checked(ref, new)


def test_identity_checked_rejects_empty_content_even_when_it_matches(tmp_path):
    ref = _write_results(tmp_path / "ref", [dict(session_id="s1", turn=0, reasoning="r", content="")])
    new = _write_results(tmp_path / "new", [dict(session_id="s1", turn=0, reasoning="r", content="")])
    with pytest.raises(ValueError, match="empty content"):
        hr.identity_checked(ref, new)


def test_identity_checked_still_raises_on_mismatched_turn_sets(tmp_path):
    ref = _write_results(tmp_path / "ref", [dict(session_id="s1", turn=0, reasoning="r", content="c")])
    new = _write_results(tmp_path / "new", [dict(session_id="s2", turn=0, reasoning="r", content="c")])
    with pytest.raises(ValueError, match="different turns"):
        hr.identity_checked(ref, new)
