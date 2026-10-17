"""A capture that is aborted (the wrapper's node-0 load watchdog sends SIGTERM) must take its server and the server's
children with it, and the wrapper must clear what a dead capture left. Fake servers, short sleeps, no GPU (CPU)."""

import os
import signal
import subprocess
import sys
import textwrap
import time

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

DSPARK = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "dspark")
)
REAP = os.path.join(DSPARK, "reap_group.sh")


def _gone(pgid: int, within_s: float = 10.0) -> bool:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _orphan_group(tmp_path, children: str) -> int:
    """A session whose leader has exited and whose children live on (reparented to init), as after an abort. Returns the
    group id, which is the leader's pid; records it in tmp_path/server.pgid."""
    pgfile = tmp_path / "server.pgid"
    subprocess.run(["setsid", "bash", "-c", f"echo $$ > {pgfile}; {children}; exit 0"], check=True, timeout=30)
    deadline = time.monotonic() + 10
    while not pgfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return int(pgfile.read_text())


def _reap(tmp_path, wait_s):
    return subprocess.run(
        ["bash", "-c", f'. {REAP}; reap_server_group {tmp_path / "server.pgid"} fake {wait_s}'],
        capture_output=True, text=True, timeout=60,
    )


def test_reap_terminates_an_orphaned_group(tmp_path):
    pgid = _orphan_group(tmp_path, "sleep 300 & sleep 300 &")
    assert not _gone(pgid, 0.2)
    r = _reap(tmp_path, 5)
    assert r.returncode == 0 and "ended on SIGTERM" in r.stdout and "still has members" in r.stdout, r.stdout
    assert _gone(pgid)


def test_reap_kills_a_member_that_ignores_sigterm(tmp_path):
    pgid = _orphan_group(tmp_path, "bash -c 'trap \"\" TERM; sleep 300' &")
    r = _reap(tmp_path, 1)
    assert r.returncode == 0 and "SIGKILL" in r.stdout and "gone" in r.stdout, r.stdout
    assert _gone(pgid)


def test_reap_does_nothing_for_a_missing_or_empty_group(tmp_path):
    r = _reap(tmp_path, 1)
    assert r.returncode == 0 and r.stdout == ""
    (tmp_path / "server.pgid").write_text("99999999\n")
    r = _reap(tmp_path, 1)
    assert r.returncode == 0 and r.stdout == ""


def test_server_clear_reports_a_held_gpu(tmp_path):
    for name, body in (("nvidia-smi", "echo 4242"), ("ss", "true")):
        stub = tmp_path / name
        stub.write_text(f"#!/bin/bash\n{body}\n")
        stub.chmod(0o755)
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", "-c", f". {REAP}; server_clear 30032 fake"], capture_output=True, text=True, env=env,
                       timeout=30)
    assert r.returncode == 1 and "GPU still held by pid(s) 4242" in r.stdout
    (tmp_path / "nvidia-smi").write_text("#!/bin/bash\n")
    r = subprocess.run(["bash", "-c", f". {REAP}; server_clear 30032 fake"], capture_output=True, text=True, env=env,
                       timeout=30)
    assert r.returncode == 0 and r.stdout == ""


_CAPTURE = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {dspark!r})
    import spec_margin_capture as smc

    out = sys.argv[1]
    smc.raise_on_signal()
    server = None
    interrupted = False
    try:
        with open(os.path.join(out, "server.log"), "w") as log:
            server = smc.launch_server(["bash", "-c", "sleep 300 & sleep 300 & wait"], os.environ, out, log)
        print("ready", flush=True)
        time.sleep(120)
    except BaseException:
        interrupted = True
        raise
    finally:
        if server is not None:
            smc.stop_server(server, 2 if interrupted else 300)
    """
)


def test_an_aborted_capture_takes_its_whole_server_group_with_it(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", _CAPTURE.format(dspark=DSPARK), str(tmp_path)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready"
        pgid = int((tmp_path / "server.pgid").read_text())
        assert pgid != os.getpgid(proc.pid), "the server must lead its own group"
        time.sleep(0.3)
        assert not _gone(pgid, 0.2), "the fake server and its sleeps should be running"
        proc.send_signal(signal.SIGTERM)  # what the wrapper's `kill $cap` sends
        assert proc.wait(timeout=60) == 128 + signal.SIGTERM
        assert _gone(pgid)
    finally:
        if proc.poll() is None:
            proc.kill()
