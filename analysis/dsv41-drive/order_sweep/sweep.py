"""Run each listed test file under three class orders (alpha = what CI's unittest does, file = pytest's, rev =
reverse alphabetical) and write one log per order into logs/. Compare with order_sweep_report.py.

Setup used for the recorded sweep (see order_sweep_result.txt):
  E=<a git-archive export of the tree>; put orderplug.py in $E; put shim/sitecustomize.py (below) at $E/shim;
  the files to sweep, one per line, in $E/rerun_files.txt; mkdir $E/logs2; run this with the venv python.
The three orders of ONE file run concurrently, so a test with a fixed port or shared path collides with itself and
looks order-dependent: re-run every hit serially before believing it (two of three hits here were that).
Each pytest gets its own session (start_new_session): without it the driver died repeatedly, cause not identified.
CUDA_VISIBLE_DEVICES must be non-empty: sglang/test/test_utils.py indexes it ([0]) at import, so an empty value
raises IndexError for every file that imports it; "9" (no such device) hides the GPU.

shim/sitecustomize.py (sweep-only, nothing installed in the venv):
    import pyarrow as _pa
    if not hasattr(_pa, "PyExtensionType"):  # datasets subclasses it; pyarrow >= 21 removed it
        _pa.PyExtensionType = _pa.ExtensionType
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
E = "/data/models/slang/nvfp4-work/t1-ordersweep"
PY = "/data/models/slang/.venv/bin/python"
os.chdir(E)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="9", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
           PYTHONPATH=f"{E}/shim:{E}/python:{E}", SGLANG_JIT_CACHE_DIR=f"{E}/.jit")
done = set(l.strip() for l in open("progress3.txt")) if os.path.exists("progress3.txt") else set()
todo = [l.strip() for l in open("rerun_files.txt") if l.strip() and l.strip() not in done]
print(time.strftime("%T"), "todo", len(todo), flush=True)


def one(f):
    base = f.replace("/", "_")
    procs = []
    for o in ("alpha", "file", "rev"):
        log = open(f"logs2/{base}.{o}", "w")
        p = subprocess.Popen(["nice", "-n", "10", "taskset", "-c", "0-63", "timeout", "400", PY, "-m", "pytest", "-p", "orderplug", f,
                              "-q", "-rA", "-p", "no:cacheprovider"], stdout=log, stderr=subprocess.STDOUT,
                             env=dict(env, ORDER=o), stdin=subprocess.DEVNULL, start_new_session=True)
        procs.append((p, log))
    for p, log in procs:
        code = p.wait()
        log.write(f"exit={code}\n")
        log.close()
    with open("progress3.txt", "a") as fh:
        fh.write(f + "\n")
    print(time.strftime("%T"), "done", f, flush=True)


with ThreadPoolExecutor(3) as ex:
    list(ex.map(one, todo))
open("sweep3.done", "w").write("finished\n")
print(time.strftime("%T"), "finished", flush=True)
