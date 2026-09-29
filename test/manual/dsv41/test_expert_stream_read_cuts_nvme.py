"""divix01 only (plan 2026-09-28-iopoll-read-cuts Task 6): the mirror roots' sysfs limits, and the io-wq punt check
through the real reader under IORING_SETUP_IOPOLL. Run under rowimg-disk.lock with IOPOLL_CUTS_DIR on an NVMe root.
Each phase runs in a fresh interpreter: io-wq workers belong to the process and linger after their last request, so
a shared process would carry one phase's workers into the next."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOTS = {"/mnt/nvme0": 520192, "/mnt/nvme2": 520192, "/mnt/nvme4": 262144}
LAYER = "dsv41_flash/exl3_row_images/layer-000.rows"
INCLUDE = Path(__file__).resolve().parents[3] / "python/sglang/kernels/jit/csrc"
pytestmark = pytest.mark.skipif(not all(Path(r, LAYER).exists() for r in ROOTS), reason="divix01 mirror roots only")

_LIMITS = r"""
#include "moe/expert_stream/host/read_cuts.h"
#include <fcntl.h>
#include <cstdio>
#include <unistd.h>
int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    int fd = open(argv[i], O_RDONLY | O_DIRECT);
    auto d = sglang::expert_stream::device_limits(fd);
    std::printf("%s %lld %llu %s\n", argv[i], (long long)d.cut_bytes, (unsigned long long)d.virt_mask, d.source.c_str());
    close(fd);
  }
}
"""


def test_sysfs_limits_of_the_mirror_roots(tmp_path):
    (tmp_path / "limits.cpp").write_text(_LIMITS)
    subprocess.run(["c++", "-std=c++20", "-I", str(INCLUDE), str(tmp_path / "limits.cpp"), "-o",
                    str(tmp_path / "limits")], check=True)
    out = subprocess.check_output([tmp_path / "limits", *[str(Path(r, LAYER)) for r in ROOTS]], text=True)
    for line, (root, cut) in zip(out.splitlines(), ROOTS.items()):
        path, got, mask, source = line.split(" ", 3)
        assert int(got) == cut and int(mask) == 4095 and source.endswith("/queue"), line


# One phase: build the fixture, read it N times through the reader with O_DIRECT under MODE=iopoll, and report the
# io-wq workers seen in /proc/self/task, the SQE lengths, and whether the slabs match the reference.
_PHASE = r"""
import json, os, sys, threading, time
from pathlib import Path
from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes
root = Path(sys.argv[1]); cap = int(sys.argv[2])
s = ram_miss_setup(root, capacity=8, experts=8, layers=2, row_images=True, hidden=2048, inter=1536)
# The fixture was just written through the page cache: flush it, or a NOWAIT O_DIRECT read of a range that still
# needs writeback (or holds delayed-allocation blocks) returns -EAGAIN and is punted whatever its size. The mirror
# roots' real row images are long settled.
os.sync()
seen, stop = set(), threading.Event()
def sample():
    while not stop.is_set():
        for t in os.listdir("/proc/self/task"):
            try:
                if open(f"/proc/self/task/{t}/comm").read().startswith("iou-wrk"): seen.add(t)
            except OSError: pass
        time.sleep(0.002)
th = threading.Thread(target=sample); th.start()
lengths, ok, info = [], True, {}
try:
    for _ in range(20):
        result, log, info, _ = read_rows_sqes(s.tables, 1, list(range(8)), list(range(8)),
                                              max_sqes=65536, leg_cut_cap=cap)
        ok = ok and result == 1
        lengths += [n for _, _, n, _ in log]
finally:
    stop.set(); th.join()
print(json.dumps({"ok": ok, "workers": len(seen), "max_len": max(lengths), "cut_reads": info.get("cut_reads", 0)}))
"""


def _phase(tmp, env_extra, cap=0):
    work = Path(os.environ["IOPOLL_CUTS_DIR"]) / tmp
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_EXPERT_STREAM_URING_")}
    env |= {"SGLANG_EXPERT_STREAM_URING_MODE": "iopoll", **env_extra}
    done = subprocess.run([sys.executable, "-c", _PHASE, str(work), str(cap)], env=env, text=True,
                          capture_output=True, timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.skipif("IOPOLL_CUTS_DIR" not in os.environ, reason="set IOPOLL_CUTS_DIR to a dir on an NVMe root")
def test_iopoll_punts_uncut_reads_and_never_cut_ones():
    cut = _phase("cut", {"SGLANG_EXPERT_STREAM_URING_READ_CUTS": "1"})
    uncut = _phase("uncut", {"SGLANG_EXPERT_STREAM_URING_READ_CUTS": "0"})
    # The control must punt, or the check proves nothing: its reads must exceed the device's limit.
    assert uncut["ok"] and uncut["max_len"] > 520192, f"raise hidden/inter: reads of {uncut['max_len']} B"
    assert uncut["workers"] > 0, uncut
    assert cut["ok"] and cut["cut_reads"] > 0 and cut["workers"] == 0, cut


@pytest.mark.skipif("IOPOLL_CUTS_DIR" not in os.environ, reason="set IOPOLL_CUTS_DIR to a dir on an NVMe root")
def test_a_cut_too_small_for_the_reads_is_refused():
    work = Path(os.environ["IOPOLL_CUTS_DIR"]) / "refuse"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SGLANG_EXPERT_STREAM_URING_")}
    done = subprocess.run([sys.executable, "-c", _PHASE, str(work), "4096"], env=env, text=True,
                          capture_output=True, timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    assert done.returncode != 0 and "legs, more than 255" in done.stderr, done.stderr[-2000:]
