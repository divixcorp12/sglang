"""The quick run script's shape: arms, order, sessions, preflight position, lock, F108 use (CPU, reads text)."""

import os
import re

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SCRIPT = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "run_l2_t3_quick.sh")


def _text():
    with open(SCRIPT) as f:
        return f.read()


def _ab_line():
    (line,) = [ln for ln in _text().splitlines() if "both_cpu_ab.py" in ln and ln.lstrip().startswith("run ")]
    return line


def test_runs_only_l2_then_t3_on_sessions_0_to_3():
    """Mutant: swapping the arm order, adding an arm, or changing the sessions turns this red."""
    tokens = _ab_line().split()
    assert tokens[tokens.index('"$OUT/fwd"') + 1 :][:2] == ['"$L2"', '"$T3"']
    assert "--sessions 0,1,2,3" in _ab_line()
    assert "L2=dspark-both-prefetch-gpu-misscut-idle2-l2\n" in _text()
    assert "T3=dspark-both-prefetch-gpu-misscut-idle2-l2-t3\n" in _text()


def test_preflight_runs_before_the_arms_and_the_lock_is_the_disk_lock():
    """Mutant: preflight after both_cpu_ab, or the flock moved off the disk lock, turns this red."""
    text = _text()
    assert text.index("l2_t3_preflight.py") < text.index("both_cpu_ab.py\" \"$OUT/fwd\"")
    assert "DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock" in text
    assert 'exec flock "$DISK_LOCK"' in text
    assert "cc-gpu.lock" not in text.replace("takes cc-gpu.lock itself", "")


def test_f108_path_is_only_baseline_and_f108_argument():
    """Mutant: passing the F108 dir as an out dir or arm arg (a write into F108) turns this red."""
    text = _text()
    uses = [
        ln.strip()
        for ln in text.splitlines()
        if "$F108_FWD" in ln and not ln.startswith(("F108_FWD=", "    echo"))
    ]
    assert len(uses) == 3  # the -d existence guard, --baseline, --f108
    assert any(re.search(r'--baseline "\$F108_FWD"', ln) for ln in uses)
    assert any(re.search(r'--f108 "\$F108_FWD"', ln) for ln in uses)
    assert "F108_FWD=$ROOT/l2-ab-20261012/fwd" in text
    assert "record_commit \"$F108_FWD" not in text
