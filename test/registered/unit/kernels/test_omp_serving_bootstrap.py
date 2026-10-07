"""Fresh-process and multiprocessing-spawn checks for the diagnostic-only hook."""
import json
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[4]
BOOTSTRAP = ROOT / "test/manual/dsv41/omp_bootstrap"
LIBRARY = os.environ.get("DSV41_OMP_TEST_LIBRARY")
pytestmark = pytest.mark.skipif(sys.platform != "linux" or not LIBRARY,
                                reason="requires the audited remote OpenMP binary")


def environment(tmp_path):
    return {**os.environ, "PYTHONPATH": str(BOOTSTRAP),
            "DSV41_OMP_INIT_CPUS": ",".join(map(str, range(64))),
            "DSV41_OMP_LIBRARY": LIBRARY,
            "DSV41_OMP_MANIFEST_DIR": str(tmp_path / "manifests")}


def test_spawn_inherits_budget_and_restores_affinity(tmp_path):
    script = tmp_path / "spawn_probe.py"
    script.write_text('''
import multiprocessing, os
def child():
    assert os.sched_getaffinity(0) == {6}
if __name__ == '__main__':
    assert os.sched_getaffinity(0) == {6}
    p = multiprocessing.get_context('spawn').Process(target=child)
    p.start(); p.join(10)
    assert p.exitcode == 0
''')
    subprocess.run(["taskset", "-c", "6", sys.executable, str(script)],
                   env=environment(tmp_path), check=True, timeout=20)
    data = [json.loads(p.read_text()) for p in (tmp_path / "manifests").glob("*.json")]
    # Parent, spawn worker and multiprocessing resource tracker all initialize.
    assert len(data) >= 2
    assert all(row["counters"]["available_cpus"] == 64 and
               row["restored_affinity"] == [6] for row in data)


def test_hash_mismatch_fails_closed(tmp_path):
    library = tmp_path / "wrong.so"
    library.write_bytes(b"different binary")
    env = environment(tmp_path)
    env["DSV41_OMP_LIBRARY"] = str(library)
    result = subprocess.run([sys.executable, "-c", "raise AssertionError('must not run')"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 78
    assert "OpenMP hash differs" in result.stderr
    assert not (tmp_path / "manifests").exists()


def test_disabled_hook_does_not_load_runtime(tmp_path):
    env = environment(tmp_path)
    env.pop("DSV41_OMP_INIT_CPUS")
    subprocess.run([sys.executable, "-c",
                    "from pathlib import Path; assert 'libgomp' not in Path('/proc/self/maps').read_text()"],
                   env=env, check=True, timeout=10)
    assert not (tmp_path / "manifests").exists()


def test_sampler_ignores_exited_scheduler_with_empty_maps(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "omp_capture", ROOT / "benchmarks/dsv41_baseline/run_omp_serving_capture.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    child = subprocess.Popen([sys.executable, "-c", """
import ctypes, os
ctypes.CDLL(None).prctl(15, b'sglang::sched', 0, 0, 0)
os._exit(0)
"""], env=environment(tmp_path))
    try:
        proc = Path('/proc') / str(child.pid)
        deadline = time.monotonic() + 10
        while proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert proc.joinpath('maps').read_text() == ''
        output = io.StringIO()
        capture.sample_runtime(tmp_path / 'manifests', output)
        assert output.getvalue() == ''
    finally:
        child.wait(timeout=10)
