"""run_capture_pair.sh's --dry-run: each arm's resolved server env, capture command and output paths, nothing
launched (CPU)."""

import json
import os
import subprocess
import sys

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SCRIPT = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "run_capture_pair.sh")
BASE = "dspark-both-misscut"


def _dry(tmp_path, *args, env=None):
    return subprocess.run(["bash", SCRIPT, str(tmp_path / "out"), "--arm", BASE, "--python", sys.executable,
                           "--dry-run", *args], capture_output=True, text=True, env=os.environ | (env or {}), timeout=300)


def test_a_dry_run_of_two_arms_resolves_their_envs_commands_and_outputs_and_launches_nothing(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("a prompt")
    out = tmp_path / "out"
    capture = ["--no-verify-split", "--long-prompt", str(prompt), "--decode-seconds", "300", "--ttft-repeats", "3"]
    r = _dry(tmp_path, "--env", "SGLANG_DSV41_CPU_EXPERTS_SPLIT=0,1,2", "ab-64=SGLANG_DSV41_FORCE_PREFILL_SHARE_ROWS=64",
             "ab-start=", "--", *capture, env={"SGLANG_LEAKED_FROM_SHELL": "1"})
    assert r.returncode == 0, r.stderr + r.stdout
    lines = r.stdout.splitlines()
    assert "DRY RUN: nothing launched" in lines and not out.exists()
    assert "unset SGLANG_LEAKED_FROM_SHELL (from the shell)" in lines
    resolved = json.loads(next(x for x in lines if x.startswith("resolved-env.json: ")).split(": ", 1)[1])
    assert set(resolved) == {"ab-64", "ab-start", "argv"}
    a, b = resolved["ab-64"], resolved["ab-start"]
    assert a["SGLANG_DSV41_FORCE_PREFILL_SHARE_ROWS"] == "64" and "SGLANG_DSV41_FORCE_PREFILL_SHARE_ROWS" not in b
    assert a["SGLANG_DSV41_CPU_EXPERTS_SPLIT"] == b["SGLANG_DSV41_CPU_EXPERTS_SPLIT"] == "0,1,2"
    assert "SGLANG_LEAKED_FROM_SHELL" not in a
    assert a["SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX"] == f"{out}/ab-64/events"
    assert b["SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX"] == f"{out}/ab-start/events"
    assert a["SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY"] == "1048576"
    assert "SGLANG_DSV41_EXPERT_TRACE_PATH" not in a
    assert resolved["argv"][resolved["argv"].index("--port") + 1] == "30032"
    for name, extra in (("ab-64", "SGLANG_DSV41_FORCE_PREFILL_SHARE_ROWS=64"), ("ab-start", "")):
        i = lines.index(f"ARM {name}")
        env_line, prelaunch, cap, outputs = lines[i + 1:i + 5]
        assert env_line.split() == ["env:", "SGLANG_DSV41_CPU_EXPERTS_SPLIT=0,1,2", *extra.split()]
        assert f"--log {out}/{name}.prelaunch.json --hicache-gib 12.67" in prelaunch
        assert cap.endswith(f"analysis/dsv41-drive/dspark/spec_margin_capture.py {out}/{name} --arm {BASE} "
                            + " ".join(capture))
        assert outputs.split()[1:5] == [f"{out}/{name}", f"{out}/{name}.memsample.jsonl", f"{out}/{name}.nvme3n1.samples",
                                        f"{out}/{name}.gpu-mem.csv"]


def test_an_arm_holding_the_engram_scales_on_node_1_gets_their_size_on_its_floor(tmp_path):
    """Each arm's prelaunch floor takes its own --extra-gib from its resolved env: the Engram scale rows' 5.72 GiB
    with SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE=1, none without it or on node 0."""
    r = _dry(tmp_path, "off=", "on=SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE=1", "node0=SGLANG_DSV41_ENGRAM_SCALES_RAM_NODE=0")
    assert r.returncode == 0, r.stderr + r.stdout
    lines = r.stdout.splitlines()
    for name, extra in (("off", "0.0"), ("on", "5.72"), ("node0", "0.0")):
        prelaunch = lines[lines.index(f"ARM {name}") + 2]
        assert prelaunch.endswith(f"--hicache-gib 12.67 --extra-gib {extra}"), prelaunch


def test_a_single_arm_with_the_default_probe_is_a_diagnostic(tmp_path):
    r = _dry(tmp_path, "diag=")
    assert r.returncode == 0, r.stderr + r.stdout
    resolved = json.loads(next(x for x in r.stdout.splitlines() if x.startswith("resolved-env.json: ")).split(": ", 1)[1])
    assert set(resolved) == {"diag", "argv"}
    assert resolved["diag"]["SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY"] == "524288"


def test_an_arm_setting_the_capture_would_override_is_refused(tmp_path):
    r = _dry(tmp_path, "a=SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY=8")
    assert r.returncode == 2 and "a: SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY" in r.stderr


def test_malformed_arms_and_missing_arguments_are_refused(tmp_path):
    for args in (["a=NOT-AN-ASSIGNMENT"], ["a=", "a="], ["bad/name="], []):
        r = _dry(tmp_path, *args)
        assert r.returncode == 2, args
    r = subprocess.run(["bash", SCRIPT, str(tmp_path / "out"), "--dry-run", "a="], capture_output=True, text=True)
    assert r.returncode == 2 and "no --arm" in r.stderr
