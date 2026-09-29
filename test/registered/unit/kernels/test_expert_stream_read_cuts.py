"""Device-sized read cuts in the expert-stream reader (plan 2026-09-28-iopoll-read-cuts). The native part compiles
read_cuts.h alone; the FFI part (Tasks 4-5) reads through the reader with a test-only cut cap (fault word
leg_cut_cap). Why the cuts exist: analysis/dsv41-drive/iopoll/diagnosis.md."""

import errno
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes, read_rows_with_fault
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
INCLUDE = ROOT / "python/sglang/kernels/jit/csrc"

_UNIT = r"""
#include "moe/expert_stream/host/read_cuts.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
using namespace sglang::expert_stream;

static DeviceLimits lim(int64_t cut, uint64_t mask) { return DeviceLimits{cut, mask, "test"}; }

struct Plan { std::vector<CutLeg> legs; std::vector<iovec> iov; unsigned n; };
static Plan plan(std::vector<iovec> in, const DeviceLimits& l, unsigned max_legs = 64) {
  Plan p; p.legs.resize(max_legs); p.iov.resize(in.size() + max_legs);
  p.n = cut_legs(in.data(), in.size(), l, p.iov.data(), p.iov.size(), p.legs.data(), max_legs);
  return p;
}
// The legs tile the input: same bytes, same addresses, in order, every leg within the cut.
static void tiles(const std::vector<iovec>& in, const Plan& p, int64_t cut) {
  size_t i = 0, off = 0, k = 0;
  for (unsigned g = 0; g < p.n; ++g) {
    assert(p.legs[g].first == k && p.legs[g].bytes > 0 && p.legs[g].bytes <= cut);
    int64_t sum = 0;
    for (unsigned j = 0; j < p.legs[g].count; ++j, ++k) {
      assert(p.iov[k].iov_base == static_cast<uint8_t*>(in[i].iov_base) + off);
      sum += p.iov[k].iov_len; off += p.iov[k].iov_len;
      if (off == in[i].iov_len) { ++i; off = 0; }
    }
    assert(sum == p.legs[g].bytes);
  }
  assert(i == in.size());
}

int main(int, char** argv) {
  // cut_bytes_for: max_sectors_kb, and max_segments less one page (a leg that starts mid-page spans one more).
  assert(cut_bytes_for(512, 128) == 520192);   // Samsung 990 EVO Plus on divix01
  assert(cut_bytes_for(256, 65) == 262144);    // SPCC on divix01
  assert(cut_bytes_for(1024, 33) == 131072);
  assert(cut_bytes_for(1, 128) == 4096);       // never below one page
  assert(cut_bytes_for(10, 0) == 8192);        // max_segments unknown: max_sectors_kb only, whole pages

  uint8_t* a = static_cast<uint8_t*>(std::aligned_alloc(4096, 1 << 20));
  uint8_t* b = static_cast<uint8_t*>(std::aligned_alloc(4096, 1 << 20));
  const int64_t C = 8192;
  { // size only: 3C + 512 in one page-aligned iovec -> C, C, C, 512
    std::vector<iovec> in{{a, 3 * C + 512}};
    Plan p = plan(in, lim(C, 4095)); assert(p.n == 4); tiles(in, p, C);
    for (unsigned g = 0; g < p.n; ++g) assert(!p.legs[g].gap);
  }
  { // gap: the first iovec ends off the page (9216 B) -> a new leg at the join even under a huge cut
    std::vector<iovec> in{{a, 9216}, {b, 4096}};
    Plan p = plan(in, lim(1 << 30, 4095)); assert(p.n == 2 && !p.legs[0].gap && p.legs[1].gap); tiles(in, p, 1 << 30);
  }
  { // aligned join: one leg
    std::vector<iovec> in{{a, 8192}, {b, 4096}};
    Plan p = plan(in, lim(1 << 30, 4095)); assert(p.n == 1 && p.legs[0].count == 2);
  }
  { // the next iovec starts off the page -> gap
    std::vector<iovec> in{{a, 8192}, {b + 512, 4096}};
    Plan p = plan(in, lim(1 << 30, 4095)); assert(p.n == 2 && p.legs[1].gap);
  }
  { // a virtually contiguous join (the next iovec starts where the last ended) is no gap, even off the page
    std::vector<iovec> in{{a, 9216}, {a + 9216, 4096}};
    Plan p = plan(in, lim(1 << 30, 4095)); assert(p.n == 1 && p.legs[0].count == 2); tiles(in, p, 1 << 30);
  }
  { // no virt boundary (mask 0): no gap rule
    std::vector<iovec> in{{a, 9216}, {b + 512, 4096}};
    Plan p = plan(in, lim(1 << 30, 0)); assert(p.n == 1);
  }
  { // gap and size together: 9216 -> C, 1024 | gap | 3C -> C, C, C
    std::vector<iovec> in{{a, 9216}, {b, 3 * C}};
    Plan p = plan(in, lim(C, 4095)); assert(p.n == 5); tiles(in, p, C);
    assert(!p.legs[0].gap && !p.legs[1].gap && p.legs[2].gap && !p.legs[3].gap && !p.legs[4].gap);
  }
  { // overflow: more legs than the caller sized
    std::vector<iovec> in{{a, 10 * C}};
    assert(plan(in, lim(C, 4095), 4).n == 5);
  }
  { // leg_bound holds over random shapes (512-aligned, the reader's alignment)
    uint64_t s = 88172645463325252ull;
    auto next = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int t = 0; t < 2000; ++t) {
      const unsigned n = 1 + next() % 6; std::vector<iovec> in; int64_t total = 0;
      uint8_t* at = a;
      for (unsigned i = 0; i < n; ++i) {
        const size_t len = 512 * (1 + next() % 200);
        const size_t skip = 512 * (next() % 9);
        at += skip; if (at + len > a + (1 << 20)) break;
        in.push_back({at, len}); total += len; at += len;
      }
      if (in.empty()) continue;
      const int64_t cut = 4096 * (1 + next() % 16);
      Plan p = plan(in, lim(cut, 4095), 255);
      assert(p.n <= leg_bound(total, in.size(), cut)); tiles(in, p, cut);
    }
  }
  { // a queue directory: values read, source named; a missing attribute is a clear refusal
    const std::string dir = argv[1];
    std::ofstream(dir + "/max_sectors_kb") << "256\n";
    std::ofstream(dir + "/max_segments") << "65\n";
    std::ofstream(dir + "/virt_boundary_mask") << "4095\n";
    DeviceLimits d; std::string why;
    assert(limits_from_queue_dir(dir, &d, &why) && d.cut_bytes == 262144 && d.virt_mask == 4095 && d.source == dir);
    std::ofstream(dir + "/chunk_sectors") << "256\n";   // a boundary the cuts do not model: named in the source
    assert(limits_from_queue_dir(dir, &d, &why) && d.cut_bytes == 262144 && d.source.find("chunk_sectors=256") != std::string::npos);
    std::remove((dir + "/chunk_sectors").c_str());
    std::remove((dir + "/virt_boundary_mask").c_str());
    assert(limits_from_queue_dir(dir, &d, &why) && d.virt_mask == 4095);   // absent: conservative 4095
    std::remove((dir + "/max_segments").c_str());
    assert(!limits_from_queue_dir(dir, &d, &why) && why.find("max_segments") != std::string::npos);
  }
  { // a file on tmpfs has no block queue: the fallback, with its reason
    const int fd = open(argv[2], O_RDONLY);
    assert(fd >= 0);
    const DeviceLimits d = device_limits(fd);
    close(fd);
    assert(d.cut_bytes == kFallbackCutBytes && d.virt_mask == kFallbackVirtMask);
    assert(d.source.rfind("fallback: ", 0) == 0);
    std::cout << "fallback source: " << d.source << "\n";
  }
  std::cout << "PASS read_cuts\n";
}
"""


def test_read_cuts_planner_and_limits(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    if not Path("/dev/shm").is_dir():
        pytest.skip("no /dev/shm (tmpfs) for the fallback case")
    source = tmp_path / "cuts.cpp"
    source.write_text(_UNIT)
    binary = tmp_path / "cuts"
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Wextra", "-I", str(INCLUDE), str(source), "-o", str(binary)],
        check=True,
    )
    queue = tmp_path / "queue"
    queue.mkdir()
    shm = Path("/dev/shm") / f"read-cuts-{os.getpid()}"
    shm.write_bytes(b"\0" * 4096)
    try:
        done = subprocess.run([binary, str(queue), str(shm)], text=True, capture_output=True, timeout=30)
    finally:
        shm.unlink()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS read_cuts" in done.stdout

PREFIX = "SGLANG_EXPERT_STREAM_URING_"
EXPERTS = [10, 3, 7, 0, 11, 5, 1, 8, 2, 9, 4]
SLOTS = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
CUT = 8192


@pytest.fixture
def uring_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(PREFIX):
            monkeypatch.delenv(key)

    def set_(**values):
        for key, value in values.items():
            monkeypatch.setenv(PREFIX + key, str(value))

    return set_


def _setup(tmp_path, images, weights=None):
    root = tmp_path / "ckpt"
    root.mkdir()
    dims = {} if images else dict(hidden=256, inter=512)
    return ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)


def _snapshot(slabs):
    if isinstance(slabs, dict):
        return {k: _snapshot(v) for k, v in slabs.items()}
    return slabs.clone()


def _equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return same_bytes(a, b)


def _read(s, **faults):
    result, log, info, record = read_rows_sqes(
        s.tables, 1, EXPERTS, SLOTS, direct=bool(s.tables.row_images), max_sqes=65536, **faults
    )
    return result, log, info, record, _snapshot(s.slabs)


def _pieces(images, on):
    return ({"piece_stream": True} | ({} if images else {"pack_workers": 2})) if on else {}


def test_cuts_off_keeps_todays_credit_and_legs(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    assert s.tables.segments.shape[0] > 1  # a multi-segment row image: a read has several iovecs
    result, _, info, _, _ = _read(s)
    assert result == 1 and info["cut_reads"] == 0 and info["min_cut_bytes"] == 0
    assert info["credit"] == 16 * 3
    assert info["leg_stride"] == 1  # a default read is one leg: no per-iovec leg storage, no >255-segment refusal


def test_credit_scales_with_the_leg_bound(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    result, _, info, _, _ = _read(s, leg_cut_cap=CUT)
    assert result == 1 and info["leg_stride"] > 1 and info["min_cut_bytes"] == CUT
    assert info["credit"] == min(32768, 16 * 3 * info["leg_stride"])


def test_explicit_depth_below_the_leg_bound_is_refused(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    uring_env(QUEUE_DEPTH=2)
    with pytest.raises(RuntimeError, match="SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=2"):
        _read(s, leg_cut_cap=CUT)


@pytest.mark.parametrize("images", [False, True], ids=["bounce", "images"])
@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_cut_reads_are_byte_identical_and_within_the_cut(tmp_path, uring_env, images, pieces):
    s = _setup(tmp_path, images, (1.0, 1.0, 1.0))
    base_result, base_log, _, base_rec, base_bytes = _read(s, **_pieces(images, pieces))
    result, log, info, rec, cut_bytes = _read(s, leg_cut_cap=CUT, **_pieces(images, pieces))
    assert base_result == result == 1 and _equal(cut_bytes, base_bytes)
    assert info["cut_reads"] > 0 and all(0 < length <= CUT for _, _, length, _ in log)
    assert all(offset % 512 == 0 and length % 512 == 0 for _, offset, length, _ in log)
    assert rec["retried_bytes"] == 0 and rec["submitted_bytes"] == base_rec["submitted_bytes"]

    # The cut SQEs cover exactly the uncut ones' file ranges.
    def ranges(entries):
        spans = sorted((f, o, o + n) for f, o, n, _ in entries)
        merged = []
        for f, a, b in spans:
            if merged and merged[-1][0] == f and merged[-1][2] == a:
                merged[-1] = (f, merged[-1][1], b)
            else:
                merged.append((f, a, b))
        return merged

    assert ranges(log) == ranges(base_log)


def test_cuts_are_off_by_default_outside_iopoll(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    _, base_log, _, _, _ = _read(s)
    uring_env(MODE="default", READ_CUTS="auto")
    result, log, info, _, _ = _read(s)
    assert result == 1 and info["cut_reads"] == info["gap_cuts"] == 0 and sorted(log) == sorted(base_log)


def test_read_cuts_on_uses_the_files_device_limits(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    _, _, _, _, base_bytes = _read(s)
    uring_env(READ_CUTS=1)
    result, log, info, _, cut_bytes = _read(s)
    assert result == 1 and _equal(cut_bytes, base_bytes)
    assert info["min_cut_bytes"] >= 4096 and all(length <= info["min_cut_bytes"] for _, _, length, _ in log)


def test_gap_cuts_at_slab_rows_off_the_page(tmp_path, uring_env):
    s = _setup(tmp_path, True, (1.0, 1.0, 1.0))
    off_page = [name for layer in s.slabs.values() for name, t in layer.items()
                if (t.numel() * t.element_size() // t.shape[0]) % 4096]
    assert off_page, "the fixture has no slab row off the page; gap cuts are untested"
    _, _, _, _, base_bytes = _read(s)
    result, _, info, _, cut_bytes = _read(s, leg_cut_cap=1 << 30)   # a huge cut: only gaps cut
    assert result == 1 and info["gap_cuts"] > 0 and _equal(cut_bytes, base_bytes)


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_cut_legs_completing_out_of_order(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    result, _, info, _, cut_bytes = _read(s, leg_cut_cap=CUT, reverse_cqes=True, **_pieces(True, pieces))
    assert result == 1 and info["cut_reads"] > 0 and _equal(cut_bytes, base_bytes)


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_a_held_cut_leg_keeps_its_read_unretired_and_its_pieces_unpublished(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s, **_pieces(True, pieces))
    result, _, _, rec, cut_bytes = _read(s, leg_cut_cap=CUT, hold_ordinal=0, leg=1, **_pieces(True, pieces))
    assert result == 1 and _equal(cut_bytes, base_bytes)
    if pieces:
        assert rec["pieces_published"] == len(EXPERTS) * 8 and rec["piece_publish_refused"] == 0


def test_one_short_cut_leg_resubmits_only_that_leg(tmp_path, uring_env):
    s = _setup(tmp_path, True)
    _, clean_log, _, _, base_bytes = _read(s, leg_cut_cap=CUT)
    result, log, _, rec, cut_bytes = _read(s, leg_cut_cap=CUT, part=0, part_short=512, leg=1)
    assert result == 1 and _equal(cut_bytes, base_bytes)
    assert len(log) == len(clean_log) + 1
    extra = sorted(set(log) - set(clean_log))
    assert len(extra) == 1 and rec["retried_bytes"] == extra[0][2]


@pytest.mark.parametrize("pieces", [False, True], ids=["whole", "pieces"])
def test_one_failing_cut_leg_fails_the_read_once_after_every_leg_is_reaped(tmp_path, uring_env, pieces):
    s = _setup(tmp_path, True)
    stats, cqes = {}, []
    first, then = read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=True, part=0, part_error=errno.EIO,
        ordinal=0, leg=1, leg_cut_cap=CUT, stats=stats, cqes=cqes, **_pieces(True, pieces))
    assert (first, then) == (0, 1)
    assert stats["unfinished_jobs"] == 0 and stats["cut_reads"] > 0


@pytest.mark.parametrize("submit_first", [False, True], ids=["unconsumed", "in_flight"])
def test_ring_reset_with_cut_legs(tmp_path, uring_env, submit_first):
    s = _setup(tmp_path, True)
    first, then = read_rows_with_fault(
        s.tables, 1, EXPERTS[:4], SLOTS[:4], EXPERTS[4:], SLOTS[4:], direct=True,
        submit_error=errno.EIO, submit_call=1, submit_first=submit_first, leg_cut_cap=CUT)
    assert (first, then) == (0, 1)


@pytest.mark.parametrize("read_mode", ["fixed", "readv_fixed"])
def test_cuts_compose_with_the_fixed_fan_out(tmp_path, uring_env, read_mode):
    s = _setup(tmp_path, True)
    _, _, _, _, base_bytes = _read(s)
    uring_env(READ_MODE=read_mode)
    try:
        result, log, info, _, cut_bytes = _read(s, leg_cut_cap=CUT, fixed_chunk_cap=64 * 1024)
    except RuntimeError as e:
        if "unsupported by the running kernel" in str(e) or "requires liburing 2.10" in str(e):
            pytest.skip(str(e))
        raise
    assert result == 1 and _equal(cut_bytes, base_bytes)
    assert info["cut_reads"] > 0 and info["fixed_cuts"] > 0 and all(length <= CUT for _, _, length, _ in log)
