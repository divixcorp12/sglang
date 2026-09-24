"""CPU-only tests for the DSV4.1 baseline harness: no network, no GPU, no server."""

import json
import os
import sys

import pytest

# provenance.py lives in scripts/dsv41/, shared with (and identical to) Task 1's harness.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "dsv41"))

import arm_env
import client_latency
import clock_ramp
import compile_watch
import generations
import metrics
import paired
import provenance
import report_builder
import results_gate
import session_subset
import synthetic_corpus
import task1_verdict
import tenancy
import verdict


# --- tok/s computation (rule 1) ---


def test_decode_tokens_per_second_matches_the_driver_formula():
    assert metrics.decode_tokens_per_second(completion_tokens=5, ttft=1.0, total=3.0) == 2.0


def test_decode_tokens_per_second_none_below_two_completion_tokens():
    assert metrics.decode_tokens_per_second(completion_tokens=1, ttft=0.1, total=0.5) is None
    assert metrics.decode_tokens_per_second(completion_tokens=0, ttft=0.1, total=0.5) is None


def test_decode_tokens_per_second_none_for_nonpositive_decode_interval():
    assert metrics.decode_tokens_per_second(completion_tokens=5, ttft=3.0, total=3.0) is None


# --- median aggregation (descriptive only, never the arm-comparison statistic) ---


def test_median_tok_s_is_the_median_not_the_mean():
    values = [1.0, 2.0, 100.0]
    assert metrics.median_tok_s(values) == 2.0
    assert metrics.median_tok_s(values) != sum(values) / len(values)


def test_median_tok_s_rejects_empty_input():
    with pytest.raises(ValueError):
        metrics.median_tok_s([])


# --- sign test ---


def test_sign_test_clean_sweep_over_8_sessions():
    assert metrics.one_sided_sign_test(wins=8, n=8) == pytest.approx(1 / 256)


def test_sign_test_even_split_is_not_significant():
    assert metrics.one_sided_sign_test(wins=4, n=8) > 0.4


def test_sign_test_rejects_out_of_range_wins():
    with pytest.raises(ValueError):
        metrics.one_sided_sign_test(wins=9, n=8)


# --- session subset ---


def test_expected_session_ids_match_n_sessions():
    assert len(session_subset.EXPECTED_SESSION_IDS) == session_subset.N_SESSIONS == 8


def test_baseline_4_session_subset_is_the_first_4_and_unchanged():
    assert session_subset.BASELINE_4_SESSION_IDS == session_subset.EXPECTED_SESSION_IDS[:4]


def test_warmup_session_is_not_one_of_the_8_timed_sessions():
    assert session_subset.WARMUP_SESSION_ID not in session_subset.EXPECTED_SESSION_IDS


def test_load_expected_sessions_matches_pinned_order(tmp_path):
    corpus = tmp_path / "sessions.jsonl"
    ids = ["s0", "s1", "s2", "s3"]
    with open(corpus, "w") as f:
        for sid in ids:
            f.write(json.dumps({"session_id": sid, "turns": ["x"]}) + "\n")
    assert session_subset.load_expected_sessions(str(corpus), n=3, skip=1) == ["s1", "s2", "s3"]


def test_verify_corpus_checksum_rejects_a_mismatch(tmp_path):
    corpus = tmp_path / "sessions.jsonl"
    corpus.write_text('{"session_id": "x"}\n')
    with pytest.raises(session_subset.CorpusChecksumError):
        session_subset.verify_corpus_checksum(str(corpus), expected="0" * 64)


def test_verify_corpus_checksum_accepts_a_match(tmp_path):
    corpus = tmp_path / "sessions.jsonl"
    corpus.write_bytes(b"hello\n")
    import hashlib

    expected = hashlib.sha256(b"hello\n").hexdigest()
    assert session_subset.verify_corpus_checksum(str(corpus), expected=expected) == expected


# --- synthetic corpus (real text, truncated, context-budget asserted) ---


def _word_tokenize(text):
    return text.split()


def _word_detokenize(tokens):
    return " ".join(tokens)


def test_build_synthetic_session_truncates_to_prompt_tokens():
    session = {"session_id": "s0", "split": "val", "turns": ["one two three four five"]}
    out = synthetic_corpus.build_synthetic_session(
        session, tokenize=_word_tokenize, detokenize=_word_detokenize, prompt_tokens=3, new_tokens=2, context_length=100
    )
    assert out["turns"] == ["one two three"]
    assert out["session_id"] == "s0"
    assert out["expected"] == [None]
    assert out["source_prompt_tokens"] == 3


def test_assert_fits_context_passes_within_budget():
    synthetic_corpus.assert_fits_context(256, new_tokens=128, context_length=4096)


def test_assert_fits_context_raises_over_budget():
    with pytest.raises(synthetic_corpus.ContextBudgetError):
        synthetic_corpus.assert_fits_context(4000, new_tokens=128, context_length=4096)


def test_build_synthetic_session_raises_before_sending_when_over_budget():
    session = {"session_id": "s0", "turns": ["one two three four five"]}
    with pytest.raises(synthetic_corpus.ContextBudgetError):
        synthetic_corpus.build_synthetic_session(
            session, tokenize=_word_tokenize, detokenize=_word_detokenize,
            prompt_tokens=5, new_tokens=10, context_length=10,
        )


def test_write_jsonl_round_trips(tmp_path):
    sessions = [{"session_id": "a", "turns": ["x"]}, {"session_id": "b", "turns": ["y"]}]
    path = tmp_path / "out.jsonl"
    synthetic_corpus.write_jsonl(str(path), sessions)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines == sessions


# --- env verification (rule 4) ---


def test_verify_env_passes_when_actual_matches_expected():
    tenancy.verify_env(actual={"A": "1", "B": "2"}, expected={"A": "1"})


def test_verify_env_raises_on_a_silently_unset_flag():
    with pytest.raises(tenancy.EnvVerificationError):
        tenancy.verify_env(actual={}, expected={"SGLANG_MOE_HOT_ASYNC_PROMOTIONS": "1"})


def test_parse_proc_environ_splits_nul_separated_kv():
    raw = b"A=1\x00B=two=parts\x00\x00"
    assert tenancy.parse_proc_environ(raw) == {"A": "1", "B": "two=parts"}


# --- result gate (rule 5) ---


def test_result_gate_passes_on_exactly_8_clean_records(tmp_path):
    path = tmp_path / "results.jsonl"
    with open(path, "w") as f:
        for i in range(8):
            f.write(json.dumps({"session_id": f"s{i}"}) + "\n")
    results_gate.check_result_gate(results_gate.load_results(str(path)))


def test_result_gate_rejects_wrong_record_count(tmp_path):
    path = tmp_path / "results.jsonl"
    with open(path, "w") as f:
        for i in range(7):
            f.write(json.dumps({"session_id": f"s{i}"}) + "\n")
    with pytest.raises(results_gate.ResultGateError):
        results_gate.check_result_gate(results_gate.load_results(str(path)))


def test_result_gate_rejects_any_error_record(tmp_path):
    path = tmp_path / "results.jsonl"
    with open(path, "w") as f:
        for i in range(7):
            f.write(json.dumps({"session_id": f"s{i}"}) + "\n")
        f.write(json.dumps({"session_id": "s7", "error": "timeout"}) + "\n")
    with pytest.raises(results_gate.ResultGateError):
        results_gate.check_result_gate(results_gate.load_results(str(path)))


# --- tenancy refusal ---


def _tenancy(**overrides):
    base = dict(memory_used_mib=1000, production_running=False, sm_clock_mhz=2900, sm_clock_limit_mhz=3135)
    base.update(overrides)
    return tenancy.parse_tenancy(base)


def test_tenancy_compatible_when_identical():
    assert tenancy.tenancy_compatible(_tenancy(), _tenancy())


def test_tenancy_incompatible_on_production_state_change():
    assert not tenancy.tenancy_compatible(_tenancy(production_running=False), _tenancy(production_running=True))


def test_tenancy_incompatible_on_large_memory_delta():
    assert not tenancy.tenancy_compatible(
        _tenancy(memory_used_mib=1000), _tenancy(memory_used_mib=30000)
    )


def test_tenancy_tolerates_small_memory_noise():
    assert tenancy.tenancy_compatible(_tenancy(memory_used_mib=1000), _tenancy(memory_used_mib=1200))


# --- SM clock stability gate and cross-arm clock-profile refusal ---


def test_is_stable_false_with_fewer_than_2_samples():
    assert not clock_ramp.is_stable([])
    assert not clock_ramp.is_stable([2955])


def test_is_stable_true_when_samples_agree():
    assert clock_ramp.is_stable([2570, 2572, 2571])


def test_is_stable_false_across_the_idle_to_rest_swing():
    # The smoke's own observed spread: ~2572 under decode vs ~2947-2970 at rest.
    assert not clock_ramp.is_stable([2572, 2960])


def test_is_stable_only_considers_the_samples_given():
    # A caller passing just the last two rounds' samples, per run_arm.sh's window.
    assert clock_ramp.is_stable([2570, 2571])


def test_clock_profiles_compatible_when_close():
    assert clock_ramp.clock_profiles_compatible([2950, 2960, 2955], [2955, 2945, 2950])


def test_clock_profiles_incompatible_across_the_idle_to_ramped_swing():
    # 2570 (idle) vs 2955 (ramped): exactly the swing that invalidated a real run.
    assert not clock_ramp.clock_profiles_compatible([2570] * 8, [2955] * 8)


def test_clock_profiles_compatible_rejects_empty_input():
    with pytest.raises(ValueError):
        clock_ramp.clock_profiles_compatible([], [2955])


# --- compile-watch: detecting Triton/CUDA JIT compilation after the server is "ready" ---


def test_count_compile_events_matches_the_observed_log_line():
    text = (
        "[2026-09-21 19:32:45] Triton kernel '_hc_mix_reduce_sinkhorn_kernel' took 34.14 s "
        "to compile after serving started. Serving-time compilation can stall the engine."
    )
    assert compile_watch.count_compile_events(text) == 1


def test_count_compile_events_counts_multiple_occurrences():
    text = "took 1.0 s to compile after serving started\ntook 2.0 s to compile after serving started\n"
    assert compile_watch.count_compile_events(text) == 2


def test_count_compile_events_zero_on_clean_log():
    text = "[2026-09-21 19:33:23] The server is fired up and ready to roll!\n"
    assert compile_watch.count_compile_events(text) == 0


def test_compile_events_in_range_only_counts_the_window(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("before: took 1.0 s to compile after serving started\n")
    start = compile_watch.log_size(str(log))
    with open(log, "a") as f:
        f.write("during: took 2.0 s to compile after serving started\n")
    end = compile_watch.log_size(str(log))
    with open(log, "a") as f:
        f.write("after: took 3.0 s to compile after serving started\n")
    assert compile_watch.compile_events_in_range(str(log), start_byte=start, end_byte=end) == 1
    assert compile_watch.compile_events_in_range(str(log), start_byte=0) == 3
    assert compile_watch.compile_events_in_range(str(log), start_byte=0, end_byte=start) == 1


# --- paired comparison: per-session only, refuses tenancy, clock, or compile mismatch ---


def _write_arm(
    tmp_path, name, *, tok_s_by_session, tenancy_fields, clock_mhz=2955, session_ids=None, contaminated_ids=()
):
    arm_dir = tmp_path / name
    arm_dir.mkdir()
    ids = session_ids or list(session_subset.EXPECTED_SESSION_IDS[: len(tok_s_by_session)])
    with open(arm_dir / "results.jsonl", "w") as f:
        for sid, tok_s in zip(ids, tok_s_by_session):
            f.write(json.dumps({"session_id": sid, "decode_tokens_per_sec": tok_s, "ttft": 60.0}) + "\n")
    with open(arm_dir / "clocks.jsonl", "w") as f:
        for sid in ids:
            f.write(json.dumps({"session_id": sid, "clock_sm_start_mhz": clock_mhz, "clock_sm_end_mhz": clock_mhz}) + "\n")
    with open(arm_dir / "compile.jsonl", "w") as f:
        for sid in ids:
            f.write(
                json.dumps(
                    {"session_id": sid, "compiled_during_session": sid in contaminated_ids, "compile_events": 0}
                )
                + "\n"
            )
    manifest = {"tenancy_start": tenancy_fields, "tenancy_end": tenancy_fields}
    (arm_dir / "run-manifest.json").write_text(json.dumps(manifest))
    return str(arm_dir)


_TENANCY_FIELDS = {"memory_used_mib": 1000, "production_running": False, "sm_clock_mhz": 2900, "sm_clock_limit_mhz": 3135}


def test_paired_compare_refuses_mismatched_tenancy(tmp_path):
    tok_s = [2.0] * 8
    a = _write_arm(tmp_path, "a", tok_s_by_session=tok_s, tenancy_fields=_TENANCY_FIELDS)
    b = _write_arm(
        tmp_path, "b", tok_s_by_session=tok_s,
        tenancy_fields={**_TENANCY_FIELDS, "production_running": True},
    )
    with pytest.raises(paired.TenancyMismatchError):
        paired.compare(a, b)


def test_paired_compare_refuses_mismatched_clock_profile(tmp_path):
    tok_s = [2.0] * 8
    a = _write_arm(tmp_path, "a", tok_s_by_session=tok_s, tenancy_fields=_TENANCY_FIELDS, clock_mhz=2570)
    b = _write_arm(tmp_path, "b", tok_s_by_session=tok_s, tenancy_fields=_TENANCY_FIELDS, clock_mhz=2955)
    with pytest.raises(paired.ClockProfileMismatchError):
        paired.compare(a, b)


def test_paired_compare_reports_the_headline_statistics(tmp_path):
    a = _write_arm(tmp_path, "a", tok_s_by_session=[2.0] * 8, tenancy_fields=_TENANCY_FIELDS)
    b = _write_arm(tmp_path, "b", tok_s_by_session=[3.0] * 8, tenancy_fields=_TENANCY_FIELDS)
    result = paired.compare(a, b, a_name="A", b_name="B")
    assert result["paired_sessions"] == 8
    assert result["b_wins"] == 8
    assert result["sign_test_p_value"] == pytest.approx(1 / 256)
    assert result["delta_median"] == pytest.approx(1.0)
    assert result["ratio_median"] == pytest.approx(1.5)


def test_paired_compare_raises_on_missing_clock_samples(tmp_path):
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    with open(a_dir / "results.jsonl", "w") as f:
        for i in range(8):
            f.write(json.dumps({"session_id": f"s{i}", "decode_tokens_per_sec": 2.0}) + "\n")
    (a_dir / "clocks.jsonl").write_text("")
    (a_dir / "compile.jsonl").write_text("")
    (a_dir / "run-manifest.json").write_text(json.dumps({"tenancy_start": _TENANCY_FIELDS}))
    b = _write_arm(tmp_path, "b", tok_s_by_session=[2.0] * 8, tenancy_fields=_TENANCY_FIELDS, session_ids=[f"s{i}" for i in range(8)])
    with pytest.raises(ValueError):
        paired.compare(str(a_dir), b)


def test_paired_compare_refuses_a_compile_contaminated_session(tmp_path):
    tok_s = [2.0] * 8
    ids = list(session_subset.EXPECTED_SESSION_IDS)
    a = _write_arm(tmp_path, "a", tok_s_by_session=tok_s, tenancy_fields=_TENANCY_FIELDS, contaminated_ids=[ids[3]])
    b = _write_arm(tmp_path, "b", tok_s_by_session=tok_s, tenancy_fields=_TENANCY_FIELDS)
    with pytest.raises(paired.CompileContaminationError):
        paired.compare(a, b)


# --- provenance.py: the pid parameter added for sampling a separate server process ---


def test_process_tree_cpu_s_default_pid_matches_current_process():
    # Both should measure the same process (this one); loose bound, just confirms the plumbing.
    default = provenance.process_tree_cpu_s()
    explicit = provenance.process_tree_cpu_s(pid=os.getpid())
    assert default is not None and explicit is not None
    assert abs(default - explicit) < 1.0


def test_process_tree_cpu_s_returns_none_for_a_dead_pid():
    # A pid essentially guaranteed not to exist.
    assert provenance.process_tree_cpu_s(pid=2**30) is None


# --- task1_verdict: sha256-pinned import, no real network/filesystem access needed ---


def test_sha256_file_matches_known_content(tmp_path):
    path = tmp_path / "f.py"
    path.write_bytes(b"hello\n")
    import hashlib

    assert task1_verdict._sha256_file(str(path)) == hashlib.sha256(b"hello\n").hexdigest()


def test_load_task1_verdict_refuses_a_hash_mismatch(tmp_path):
    path = tmp_path / "task1_arm_verdict.py"
    path.write_text("X = 1\n")
    with pytest.raises(task1_verdict.Task1VerdictHashMismatchError):
        task1_verdict.load_task1_verdict(path=str(path), expected_sha256="0" * 64)


def test_load_task1_verdict_imports_on_a_hash_match(tmp_path):
    path = tmp_path / "task1_arm_verdict.py"
    path.write_text("X = 1\n")
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    module = task1_verdict.load_task1_verdict(path=str(path), expected_sha256=expected)
    assert module.X == 1


# --- generations.py: this campaign's code-generation registry ---


def test_check_registered_raises_for_an_unregistered_tree(tmp_path):
    path = tmp_path / "generations.json"
    with pytest.raises(generations.UnregisteredGenerationError):
        generations.check_registered("deadbeef", path=str(path))


def test_register_then_check_registered_round_trips(tmp_path):
    path = tmp_path / "generations.json"
    generations.register("deadbeef", "gen1", path=str(path))
    assert generations.check_registered("deadbeef", path=str(path)) == "gen1"


def test_register_persists_across_loads(tmp_path):
    path = tmp_path / "generations.json"
    generations.register("aaa", "gen1", path=str(path))
    generations.register("bbb", "gen2", path=str(path))
    manifest = generations.load(str(path))
    assert manifest["generations"] == {"aaa": "gen1", "bbb": "gen2"}


# --- report_builder: shaping HTTP results into Task 1's report schema ---


def test_merge_sessions_joins_by_session_id():
    results = [{"session_id": "s0", "decode_tokens_per_sec": 2.5, "ttft": 60.0}]
    clocks_by_id = {"s0": {"clock_sm_start_mhz": 2570, "clock_sm_end_mhz": 2580}}
    compile_by_id = {"s0": {"compiled_during_session": False, "compile_events": 0}}
    cpu_s_by_id = {"s0": 12.3}
    [merged] = report_builder.merge_sessions(
        results=results, clocks_by_id=clocks_by_id, compile_by_id=compile_by_id, cpu_s_by_id=cpu_s_by_id
    )
    assert merged["session_id"] == "s0"
    assert merged["decode_tok_s"] == 2.5
    assert merged["ttft_s"] == 60.0
    assert merged["cpu_s"] == 12.3
    assert merged["clock_sm_start_mhz"] == 2570
    assert merged["compiled_during_session"] is False
    assert "unavailable" in merged["step_latency"]


def test_merge_sessions_tolerates_missing_side_data():
    results = [{"session_id": "s0", "decode_tokens_per_sec": 2.5, "ttft": 60.0}]
    [merged] = report_builder.merge_sessions(
        results=results, clocks_by_id={}, compile_by_id={}, cpu_s_by_id={}
    )
    assert merged["clock_sm_start_mhz"] is None
    assert merged["compiled_during_session"] is None
    assert merged["cpu_s"] is None


SERVER_ENV = {
    "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
    "PYTHONPATH": "/x/python",
}


def _build(**kwargs):
    defaults = dict(
        harness_provenance={"git": {"head": "abc"}},
        server_env_actual=dict(SERVER_ENV),
        server_env_expected=dict(SERVER_ENV),
        boundary_samples=[{"label": "before_engine"}],
        residency={"dirs": [], "before": {}, "after": {}},
        sessions=[],
    )
    return report_builder.build_report(**{**defaults, **kwargs})


def test_build_report_names_the_server_provenance_ceiling():
    report = _build()
    assert report["provenance"]["git"]["head"] == "abc"
    assert "separate process" in report["provenance"]["server_provenance_ceiling"]
    assert report["boundary_samples"] == [{"label": "before_engine"}]


def test_build_report_puts_the_servers_env_where_check_arm_reads_it():
    """The bug this guards: check_arm read the harness's env tables as if they were the arm's.

    A leases-off, mirrors-on arm was graded with the harness's environment, which has none of
    these vars, and the verdict reported the reader as 'mmap' and the mirror dirs as unset for a
    server whose /proc/<pid>/environ had both set."""
    report = _build(
        harness_provenance={
            "git": {"head": "abc"},
            "sglang_env": {"SGLANG_MOE_EXPERT_FILE_READER": "mmap"},
            "sglang_env_resolved": {"SGLANG_MOE_EXPERT_FILE_READER": "mmap"},
            "sglang_file": "/harness/python/sglang/__init__.py",
        },
        server_env_actual={**SERVER_ENV, "SGLANG_MOE_EXPERT_MIRROR_DIRS": "/mnt/nvme0/x:/mnt/nvme4/x"},
    )
    prov = report["provenance"]
    assert prov["sglang_env"]["SGLANG_MOE_EXPERT_FILE_READER"] == "uring_direct"
    assert prov["sglang_env"]["SGLANG_MOE_EXPERT_MIRROR_DIRS"] == "/mnt/nvme0/x:/mnt/nvme4/x"
    # the harness's own values are kept, but not where they can be mistaken for the server's
    assert prov["harness_process"]["sglang_env"]["SGLANG_MOE_EXPERT_FILE_READER"] == "mmap"
    assert prov["harness_process"]["sglang_file"] == "/harness/python/sglang/__init__.py"


def test_build_report_reports_resolved_values_as_unavailable_not_as_the_harnesss():
    report = _build(harness_provenance={"git": {"head": "abc"}, "sglang_env_resolved": {"X": "y"}})
    prov = report["provenance"]
    assert prov["sglang_env_resolved"] is None
    assert prov["sglang_file"] is None
    assert "sglang_env_resolved" in prov["unavailable"]
    assert "sglang_file" in prov["unavailable"]


def test_build_report_keeps_an_earlier_unavailable_entry():
    report = _build(harness_provenance={"git": None, "unavailable": {"git": "sglang did not import"}})
    assert report["provenance"]["unavailable"]["git"] == "sglang did not import"
    assert "sglang_env_resolved" in report["provenance"]["unavailable"]


def test_build_report_does_not_store_the_raw_environ():
    """/proc/<pid>/environ carries whatever the launching shell held; a report gets copied."""
    report = _build(server_env_actual={
        **SERVER_ENV,
        "HF_TOKEN": "hunter2",          # not a steering knob: dropped outright
        "HOME": "/home/x",              # not a steering knob either
        "SGLANG_REMOTE_API_KEY": "sk-1",  # a knob whose name says secret: kept, redacted
    })
    for table in ("sglang_env", "server_env_actual"):
        stored = report["provenance"][table]
        assert "HF_TOKEN" not in stored
        assert "HOME" not in stored
        assert stored["SGLANG_REMOTE_API_KEY"] == "<redacted>"


# --- verdict.py: orchestrates task1_arm_verdict's functions, adds two labeled checks ---


class _FakeTask1Verdict:
    """A stand-in with task1_arm_verdict's exact call shape, so these tests exercise this
    campaign's glue code without needing the real (uncommitted, divix01-only) file."""

    def __init__(self, *, problems=None, notes=None, contended=("not contended", []), outliers=None, gen=("gen1", "tree123")):
        self._problems = problems or []
        self._notes = notes or []
        self._contended = contended
        self._outliers = outliers or []
        self._gen = gen

    def check_arm(self, report, *, root, head, mirror, traced, sessions=4):
        return list(self._problems), list(self._notes)

    def check_timed_phase(self, phases):
        return []

    def boot_growth(self, phases):
        return {d: phases["ready"][d] - phases["before"][d] for d in phases["ready"]}

    def contention(self, report):
        return self._contended

    def session_outliers(self, report):
        return list(self._outliers)

    def generation(self, report, root, manifest):
        return self._gen

    def cross_arm_outliers(self, report, arm_path, references):
        return ["CROSS-ARM session_0: decode 2.0 tok/s is 10% below the 2-arm clean median 2.2"]


def _sample_report(*, compiled_during_session=(False, False)):
    return {
        "provenance": {"git": {"head": "abc"}, "sglang_env": dict(SERVER_ENV)},
        "residency": {"dir": "/d", "before_server": 1, "server_ready": 1,
                      "after_timed_set": 1},
        "per_session": [
            {"session_id": f"s{i}", "clock_sm_start_mhz": 2570, "compiled_during_session": c, "compile_events": int(c)}
            for i, c in enumerate(compiled_during_session)
        ],
    }


def test_is_acknowledged_step_latency_problem():
    assert verdict.is_acknowledged_step_latency_problem("session 0: no step latency (fewer than two chunks)")
    assert not verdict.is_acknowledged_step_latency_problem("git head 'abc' is not the expected 'def'")


def test_server_env_problems_is_silent_on_a_correctly_configured_server():
    assert verdict.server_env_problems(_sample_report(), root="/x") == []


def test_server_env_problems_catches_the_wrong_reader():
    report = _sample_report()
    report["provenance"]["sglang_env"]["SGLANG_MOE_EXPERT_FILE_READER"] = "mmap"
    [problem] = verdict.server_env_problems(report, root="/x")
    assert "mmap" in problem and "page cache" in problem


def test_server_env_problems_separates_unset_from_wrong():
    """An unset knob is a default applied inside a process this harness cannot read. Reporting it
    as a wrong value would be the same mistake, one layer down."""
    report = _sample_report()
    del report["provenance"]["sglang_env"]["SGLANG_MOE_EXPERT_FILE_READER"]
    [problem] = verdict.server_env_problems(report, root="/x")
    assert "is unset in the server" in problem


def test_server_env_problems_catches_a_falsy_graph_gather():
    report = _sample_report()
    report["provenance"]["sglang_env"]["SGLANG_MOE_EXPERT_GRAPH_GATHER"] = "0"
    [problem] = verdict.server_env_problems(report, root="/x")
    assert "in-graph reader" in problem


def test_server_env_problems_accepts_the_other_true_spellings():
    for spelling in ("1", "true", "TRUE", "yes", "on"):
        report = _sample_report()
        report["provenance"]["sglang_env"]["SGLANG_MOE_EXPERT_GRAPH_GATHER"] = spelling
        assert verdict.server_env_problems(report, root="/x") == [], spelling


def test_server_env_problems_checks_the_tree_the_server_could_import():
    report = _sample_report()
    report["provenance"]["sglang_env"]["PYTHONPATH"] = "/other/python"
    [problem] = verdict.server_env_problems(report, root="/x")
    assert "/other/python" in problem


def test_server_env_problems_refuses_to_pass_with_no_server_env_at_all():
    report = _sample_report()
    report["provenance"].pop("sglang_env")
    [problem] = verdict.server_env_problems(report, root="/x")
    assert "none of the env checks below were made" in problem


def test_judge_acknowledges_only_the_unknowable_form_of_each_server_check():
    """check_arm's four unanswerable-over-HTTP checks are acknowledged; the measured stand-ins
    are not, so a genuinely misconfigured server still reads unacknowledged-INVALID."""
    report = _sample_report()
    report["provenance"]["sglang_env"]["SGLANG_MOE_EXPERT_FILE_READER"] = "mmap"
    fake = _FakeTask1Verdict(problems=[
        "imported sglang from None, not /x/python/sglang",
        "provenance fields unavailable: ['sglang_env_resolved', 'sglang_file']",
        "SGLANG_MOE_EXPERT_FILE_READER resolved to None, not 'uring_direct': reads may fill the page cache",
        "SGLANG_MOE_EXPERT_GRAPH_GATHER did not resolve to true: this is not the in-graph reader",
    ])
    result = verdict.judge(report, root="/x", head="abc", mirror=False, traced=False, task1_module=fake)
    assert len(result["acknowledged_problems"]) == 4
    [unacknowledged] = result["unacknowledged_problems"]
    assert "mmap" in unacknowledged
    assert result["valid_except_acknowledged_gaps"] is False


def test_judge_is_clean_when_the_server_env_answers_the_acknowledged_checks():
    fake = _FakeTask1Verdict(problems=[
        "SGLANG_MOE_EXPERT_FILE_READER resolved to None, not 'uring_direct': reads may fill the page cache",
    ])
    result = verdict.judge(_sample_report(), root="/x", head="abc", mirror=False, traced=False, task1_module=fake)
    assert result["unacknowledged_problems"] == []
    assert result["valid_except_acknowledged_gaps"] is True


def test_compile_contamination_problems_flags_only_contaminated_sessions():
    report = _sample_report(compiled_during_session=(False, True))
    problems = verdict.compile_contamination_problems(report)
    assert len(problems) == 1
    assert "s1" in problems[0]


def test_judge_reports_step_latency_as_acknowledged_not_blocking_silently():
    report = _sample_report()
    for row in report["per_session"]:
        row["step_latency"] = {"unavailable": "no per-step timing: ..."}
    fake = _FakeTask1Verdict(problems=["session 0: no step latency (...)", "session 1: no step latency (...)"])
    result = verdict.judge(report, root="/x", head="abc", mirror=False, traced=False, task1_module=fake)
    assert result["valid"] is False  # Task 1's own definition: any problem means INVALID
    assert result["valid_except_acknowledged_gaps"] is True  # nothing unacknowledged
    assert len(result["acknowledged_problems"]) == 2
    assert result["unacknowledged_problems"] == []


def test_judge_surfaces_a_real_problem_as_unacknowledged():
    report = _sample_report()
    fake = _FakeTask1Verdict(problems=["git head 'abc' is not the expected 'def'"])
    result = verdict.judge(report, root="/x", head="def", mirror=False, traced=False, task1_module=fake)
    assert result["valid"] is False
    assert result["valid_except_acknowledged_gaps"] is False
    assert result["unacknowledged_problems"] == ["git head 'abc' is not the expected 'def'"]


def test_judge_adds_compile_contamination_to_problems():
    report = _sample_report(compiled_during_session=(False, True))
    fake = _FakeTask1Verdict()
    result = verdict.judge(report, root="/x", head="abc", mirror=False, traced=False, task1_module=fake)
    assert any("JIT compilation" in p for p in result["unacknowledged_problems"])


def test_judge_includes_generation_and_contention_notes():
    report = _sample_report()
    fake = _FakeTask1Verdict()
    result = verdict.judge(
        report, root="/x", head="abc", mirror=False, traced=False, task1_module=fake, manifest={"generations": {}}
    )
    assert any(n.startswith("GENERATION gen1") for n in result["notes"])
    assert any(n.startswith("CONTENDED not contended") for n in result["notes"])


def test_judge_includes_cross_arm_notes_when_references_given():
    report = _sample_report()
    fake = _FakeTask1Verdict()
    result = verdict.judge(
        report, root="/x", head="abc", mirror=False, traced=False, task1_module=fake,
        reference_arms=["/other/arm.json"], arm_json_path="/this/arm.json",
    )
    assert any(n.startswith("CROSS-ARM") for n in result["notes"])


# --- client_latency.py: the client-side proxy, explicitly not step_latency ---


def test_client_inter_token_latency_s_computes_percentiles():
    # 5 evenly-spaced arrivals -> 4 gaps of 0.1s each.
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    result = client_latency.client_inter_token_latency_s(times)
    assert result["n"] == 4
    assert result["p50"] == pytest.approx(0.1)
    assert result["max"] == pytest.approx(0.1)


def test_client_inter_token_latency_s_unavailable_below_two_chunks():
    assert "unavailable" in client_latency.client_inter_token_latency_s([])
    assert "unavailable" in client_latency.client_inter_token_latency_s([0.5])


def test_client_inter_token_latency_s_is_not_named_step_latency():
    # A structural guard: this module's public result must never carry the "step" name,
    # so nobody greps for step_latency and finds this by accident.
    result = client_latency.client_inter_token_latency_s([0.0, 0.1, 0.2])
    assert "step" not in "".join(result.keys()).lower()


# --- report_builder: client_inter_token_latency_s is wired in, separate from step_latency ---


def test_merge_sessions_computes_client_inter_token_latency_from_chunk_times():
    results = [
        {
            "session_id": "s0",
            "decode_tokens_per_sec": 2.5,
            "ttft": 60.0,
            "chunk_times": [60.0, 60.5, 61.0],
        }
    ]
    [merged] = report_builder.merge_sessions(
        results=results, clocks_by_id={}, compile_by_id={}, cpu_s_by_id={}
    )
    assert merged["client_inter_token_latency_s"]["n"] == 2
    assert merged["client_inter_token_latency_s"]["p50"] == pytest.approx(0.5)
    # step_latency stays the acknowledged-absent placeholder, never the client proxy.
    assert "unavailable" in merged["step_latency"]
    assert merged["step_latency"] != merged["client_inter_token_latency_s"]


def test_merge_sessions_tolerates_missing_chunk_times():
    results = [{"session_id": "s0", "decode_tokens_per_sec": 2.5, "ttft": 60.0}]
    [merged] = report_builder.merge_sessions(
        results=results, clocks_by_id={}, compile_by_id={}, cpu_s_by_id={}
    )
    assert "unavailable" in merged["client_inter_token_latency_s"]


# --- verdict.py: Task 1's timed-phase residency check wired in ---


def test_residency_cache_dict_reshapes_this_campaigns_residency_section():
    report = {"residency": {"dir": "/mnt/nvme2/x", "before_server": 100, "server_ready": 100, "after_timed_set": 105}}
    cache = verdict.residency_cache_dict(report)
    assert cache == {"before": {"/mnt/nvme2/x": 100},
                     "ready": {"/mnt/nvme2/x": 100},
                     "last": {"/mnt/nvme2/x": 105}}


def test_residency_cache_dict_none_when_no_residency_section():
    assert verdict.residency_cache_dict({}) is None
    assert verdict.residency_cache_dict({"residency": {"dir": None}}) is None
    assert verdict.residency_cache_dict({"residency": {"dir": "/d", "before_server": 1,
                                                       "after_timed_set": 2}}) is None


def test_judge_calls_task1_timed_phase_with_ready_and_last():
    report = _sample_report()
    report["residency"] = {"dir": "/d", "before_server": 1, "server_ready": 2, "after_timed_set": 2}
    calls = []

    class FakeWithCache(_FakeTask1Verdict):
        def check_timed_phase(self, phases):
            calls.append(phases)
            return []

    result = verdict.judge(report, root="/x", head="abc", mirror=False, traced=False, task1_module=FakeWithCache())
    assert calls == [{"before": {"/d": 1}, "ready": {"/d": 2}, "last": {"/d": 2}}]
    assert not result["unacknowledged_problems"]
    assert any("startup" in note for note in result["notes"])


def test_judge_gates_timed_growth_not_startup_growth():
    report = _sample_report()
    report["residency"] = {"dir": "/d", "before_server": 1, "server_ready": 3 << 30,
                            "after_timed_set": 3 << 30}
    task1 = task1_verdict.load_task1_verdict(
        path=os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "dsv41-drive",
                          "task1_arm_verdict.py")
    )

    class TimedGate(_FakeTask1Verdict):
        def check_timed_phase(self, phases):
            return task1.check_timed_phase(phases)

    result = verdict.judge(report, root="/x", head="abc", mirror=False, traced=False, task1_module=TimedGate())
    assert not result["unacknowledged_problems"]

    report["residency"]["after_timed_set"] += 2 << 30
    result = verdict.judge(report, root="/x", head="abc", mirror=False, traced=False, task1_module=TimedGate())
    assert any("during the timed sessions" in problem for problem in result["unacknowledged_problems"])


@pytest.mark.parametrize("residency", [
    None,
    {"before_server": 1, "server_ready": 2, "after_timed_set": 2},
    {"dir": "/d", "server_ready": 2, "after_timed_set": 2},
    {"dir": "/d", "before_server": 1, "after_timed_set": 2},
    {"dir": "/d", "before_server": 1, "server_ready": 2, "after_timed_set": None},
])
def test_judge_missing_residency_sample_is_unacknowledged_problem(residency):
    report = _sample_report()
    if residency is None:
        del report["residency"]
    else:
        report["residency"] = residency
    result = verdict.judge(report, root="/x", head="abc", mirror=False, traced=False,
                           task1_module=_FakeTask1Verdict())
    assert any("residency" in problem for problem in result["unacknowledged_problems"])
    assert not result["valid_except_acknowledged_gaps"]


# --- arm_env.ServerArgs: decode_log_interval is opt-in, off by default ---


def test_server_args_omits_decode_log_interval_by_default():
    argv = arm_env.ServerArgs(port=31050).argv()
    assert "--decode-log-interval" not in argv


def test_server_args_includes_decode_log_interval_when_set():
    argv = arm_env.ServerArgs(port=31050, decode_log_interval=1).argv()
    i = argv.index("--decode-log-interval")
    assert argv[i + 1] == "1"


# --- arm_env: expert-row mirroring is on by default, and an arm turns it off by value ---


def test_base_env_mirrors_expert_rows_by_default():
    assert arm_env.base_env()["SGLANG_MOE_EXPERT_MIRROR_DIRS"] == arm_env.EXPERT_MIRROR_DIRS


def test_default_mirror_roots_are_two_absolute_paths_on_distinct_drives():
    roots = arm_env.EXPERT_MIRROR_DIRS.split(os.pathsep)
    assert len(roots) == 2, roots
    assert all(r.startswith("/mnt/") for r in roots), roots
    # Same drive twice would spread nothing; the point of the pair is two spindles.
    assert len({r.split("/")[2] for r in roots}) == 2, roots


def test_an_arm_turns_mirroring_off_with_an_empty_override():
    # Not by dropping the key: base_env always supplies one, so the empty string is the
    # only way off, and exl3_expert_format.exl3_mirror_config reads it as off.
    env = arm_env.arm_env({"SGLANG_MOE_EXPERT_MIRROR_DIRS": ""})
    assert env["SGLANG_MOE_EXPERT_MIRROR_DIRS"] == ""


def test_the_verdicts_mirror_flag_follows_the_value_not_the_key():
    # The expression under test is run_arm.sh's, kept in sync by hand; a key-presence
    # test would call an unmirrored arm mirrored now that base_env always sets the key.
    mirror = lambda env: bool(env.get("SGLANG_MOE_EXPERT_MIRROR_DIRS"))
    assert mirror(arm_env.base_env()) is True
    assert mirror(arm_env.arm_env({"SGLANG_MOE_EXPERT_MIRROR_DIRS": ""})) is False


# --- arm_env: the server's cores stay on NUMA node 0 (the 2026-09-22 startup hang) ---

# divix01's topology, from `numactl --hardware`. Node 1 is where the box's reth/nimbus
# stack lives, so a server thread that lands there first-touches memory into a node with
# ~9 GB free and the allocator spins in direct compaction instead of failing over.
NODE0_CPUS = frozenset(range(0, 18)) | frozenset(range(36, 54))
NODE1_CPUS = frozenset(range(18, 36)) | frozenset(range(54, 72))


def _cores(spec: str) -> set[int]:
    """Expand a taskset list like "0-7,16-17,36-53" into core numbers."""
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def test_core_spec_expansion_handles_ranges_and_singletons():
    # _cores is the test's own parser, so prove it before trusting the assertions below.
    assert _cores("3") == {3}
    assert _cores("0-2,7") == {0, 1, 2, 7}


def test_server_cores_are_entirely_on_numa_node_0():
    cores = _cores(arm_env.SERVER_CORES)
    assert cores, arm_env.SERVER_CORES
    assert cores <= NODE0_CPUS, sorted(cores - NODE0_CPUS)


def test_server_cores_touch_no_node_1_core():
    # Stated separately from the subset check: this is the property that actually caused
    # the hang, and it must fail loudly if NODE0_CPUS is ever widened by mistake.
    assert not (_cores(arm_env.SERVER_CORES) & NODE1_CPUS)


def test_server_cores_do_not_overlap_the_driver_or_the_reserved_cores():
    server = _cores(arm_env.SERVER_CORES)
    assert not (server & _cores(arm_env.DRIVER_CORES)), "server and driver share cores"
    # Core 71 is production's doorbell spin core; 64-71 stay free for every CPU job.
    assert not (server & _cores(arm_env.FREE_CORES)), "server touches the reserved cores"


def test_run_arm_pins_the_server_with_arm_envs_core_list_not_a_literal():
    # The launch line and arm_env.SERVER_CORES were two copies of "32-63" that drifted
    # apart silently; a literal core list here is the bug, not a style choice.
    script = open(os.path.join(os.path.dirname(__file__), "run_arm.sh")).read()
    launch = [l for l in script.splitlines() if "taskset" in l and "nsys_prefix" in l]
    assert len(launch) == 1, launch
    assert 'taskset -c "$server_cores"' in launch[0], launch[0]
    assert "arm_env.SERVER_CORES" in script


# --- arm_env: the pinned host buffer must fit in node 0, not in the whole box ---


def test_base_env_uses_the_declared_pinned_host_budget():
    assert arm_env.base_env()["SGLANG_MOE_PINNED_HOST_MB"] == arm_env.PINNED_HOST_MB


def test_engram_host_node_defaults_on_while_async_scores_stay_off_in_gpu_residency_mode():
    defaults = arm_env.base_env()
    assert defaults["SGLANG_MOE_ASYNC_RESIDENCY_SCORES"] == "0"
    assert defaults["SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING"] == "1"
    off_arm = arm_env.arm_env(
        {"SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING": "0"}
    )
    assert off_arm["SGLANG_MOE_ASYNC_RESIDENCY_SCORES"] == "0"
    assert off_arm["SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING"] == "0"
    assert off_arm["SGLANG_MOE_GPU_RESIDENCY_UPDATE"] == "1"
    assert off_arm["SGLANG_MOE_HOT_DYNAMIC"] == "1"


def test_pinned_buffer_and_weights_fit_in_node_0s_free_memory():
    # The arithmetic that was skipped on 2026-09-22: at 71680 MiB the pinned buffer
    # ALONE exceeded node 0's 66619 MiB free, so the arm could never have started. It
    # exhausted node 0 to 0.77 GB and spun in direct compaction for the full 900s.
    need = int(arm_env.PINNED_HOST_MB) + arm_env.WEIGHTS_AND_OVERHEAD_MIB
    assert need <= arm_env.NODE0_FREE_MIB, f"needs {need} MiB, node 0 has {arm_env.NODE0_FREE_MIB}"


def test_the_pinned_budget_is_not_silently_the_old_unbootable_value():
    # Guards the specific regression: 71680 is the value that cannot start on this box.
    assert int(arm_env.PINNED_HOST_MB) != 71680
