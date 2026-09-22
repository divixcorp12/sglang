"""init_single_process_dist must not fix the port two concurrent jobs would both bind (CPU).

The helper used to default MASTER_PORT to 29632, so a second process running any test that calls it on the same host
died with EADDRINUSE. Each case runs the helper in its own subprocess: it initialises torch.distributed and sets
environment variables, which must not leak into this process or into another test.
"""

import inspect
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

# Initialises the world-1 group, reports the port it used, and stays alive until the parent says to exit, so two of
# these are alive at once.
HOLD = textwrap.dedent(
    """
    import os, sys
    from sglang.test.layer_ut_utils import init_single_process_dist
    init_single_process_dist(**eval(sys.argv[1]))
    print("READY", os.environ["MASTER_PORT"], flush=True)
    sys.stdin.readline()
    """
)


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _spawn(kwargs="{}", env=None):
    full_env = {k: v for k, v in os.environ.items() if k != "MASTER_PORT"}
    full_env.update(env or {})
    # stderr goes to a file: an unread pipe fills with import warnings and can block the child before it says READY.
    err = tempfile.TemporaryFile("w+")
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLD, kwargs],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        env=full_env,
    )
    proc.err_file = err
    return proc


def _stderr_tail(proc):
    proc.err_file.seek(0)
    return proc.err_file.read()[-600:]


def _ready_port(proc, timeout=180):
    """The port a child reports once initialised; fails with its stderr if it died first."""
    try:
        out, _ = proc.communicate(input="\n", timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    lines = [line for line in out.splitlines() if line.startswith("READY")]
    if not lines:
        raise AssertionError(f"exit {proc.returncode}; stderr tail: {_stderr_tail(proc)}")
    return int(lines[0].split()[1])


class TestSingleProcessDistPort(unittest.TestCase):
    def test_two_concurrent_jobs_both_initialise(self):
        """Both children must be alive (holding their store) before either exits, so a shared fixed port collides."""
        first, second = _spawn(), _spawn()
        # Wait for both READY lines before releasing either.
        ready = []
        for proc in (first, second):
            line = proc.stdout.readline()
            self.assertTrue(
                line.startswith("READY"),
                f"a child failed to initialise while the other held its port: {_stderr_tail(proc)}",
            )
            ready.append(int(line.split()[1]))
        for proc in (first, second):
            proc.stdin.write("\n")
            proc.stdin.flush()
            proc.wait(timeout=60)
        self.assertNotEqual(ready[0], ready[1])

    def test_the_default_is_no_longer_a_fixed_port(self):
        self.assertIsNone(inspect.signature(init_single_process_dist).parameters["master_port"].default)

    def test_an_environment_value_that_is_already_set_wins(self):
        pinned = _free_port()
        self.assertEqual(_ready_port(_spawn(env={"MASTER_PORT": str(pinned)})), pinned)

    def test_an_explicit_port_is_used_when_the_environment_has_none(self):
        explicit = _free_port()
        self.assertEqual(_ready_port(_spawn(f"{{'master_port': {explicit}}}")), explicit)

    def test_the_environment_beats_an_explicit_port(self):
        pinned, explicit = _free_port(), _free_port()
        self.assertEqual(
            _ready_port(_spawn(f"{{'master_port': {explicit}}}", env={"MASTER_PORT": str(pinned)})), pinned
        )


if __name__ == "__main__":
    unittest.main()
