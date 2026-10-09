"""The per-verify accept log a trace run joins to its route log (CPU)."""

import json

from sglang.srt.environ import envs
from sglang.srt.speculative import verify_accept_log
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_each_verify_of_a_request_gets_its_ordinal_and_correct_draft_count(tmp_path):
    log = verify_accept_log.VerifyAcceptLog(str(tmp_path / "accept"))
    log.record("r1", 3, settled=True)
    log.record("r2", 0, settled=True)
    log.record("r1", 5, settled=False)
    log.close()
    (path,) = tmp_path.glob("accept.*.jsonl")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(e["rid"], e["k"], e["num_correct_drafts"], e["settled"]) for e in lines] == [
        ("r1", 0, 3, True), ("r2", 0, 0, True), ("r1", 1, 5, False)]
    assert all(isinstance(e["ns"], int) for e in lines)


def test_the_log_exists_only_when_its_path_is_set(tmp_path):
    verify_accept_log.reset_for_test()
    with envs.SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH.override(""):
        assert verify_accept_log.get_verify_accept_log() is None
    verify_accept_log.reset_for_test()
    with envs.SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH.override(str(tmp_path / "accept")):
        log = verify_accept_log.get_verify_accept_log()
        assert log is not None and log is verify_accept_log.get_verify_accept_log()
    verify_accept_log.reset_for_test()
