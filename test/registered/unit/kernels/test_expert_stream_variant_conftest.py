"""The expert-stream tests' instrumented default (sglang.test.expert_stream_variant) reaches every test under a conftest
that takes it, however the command line orders the files.

pytest 9.1 binds a conftest's fixtures to the first Directory node it collects for that path. A command line that lists
kernels/a.py, then a file of the parent directory, then kernels/b.py collects kernels/ twice, and the second node sees
none of the conftest's fixtures: an autouse fixture silently stops applying, and b.py's tests load the production
build. Found as test_expert_stream_drive_load.py failing "a fault ... is test-only" only after the requirements tests."""

import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CONFTEST = Path(__file__).resolve().parent / "conftest.py"
ASSERT_INSTR = """
from sglang.kernels.ops.moe import expert_stream_transport as ops


def test_the_default_is_instrumented():
    assert ops.host_variant() == "instr"
"""


def test_a_directory_revisited_on_the_command_line_keeps_the_instrumented_default(tmp_path):
    kernels = tmp_path / "unit" / "kernels"
    kernels.mkdir(parents=True)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (kernels / "conftest.py").write_text(CONFTEST.read_text())
    (kernels / "test_first.py").write_text(ASSERT_INSTR)
    (kernels / "test_again.py").write_text(ASSERT_INSTR)
    (tmp_path / "unit" / "test_between.py").write_text("def test_nothing():\n    pass\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "-p", "no:cacheprovider",
         "unit/kernels/test_first.py", "unit/test_between.py", "unit/kernels/test_again.py"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
