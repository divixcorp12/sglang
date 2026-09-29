"""A drain that discards SQEs the kernel never consumed must keep the ring and its registrations (plan
2026-09-29-ring-reset-nop-drain). Before the fix, UringReader::drain closed the ring and re-registered every buffer
chunk and file: about 59 s for the 107 GB tier on divix01 (analysis/dsv41-drive/uring-reg/results.md), past the
watchdog's 30 s fatal_wait. A native harness drives UringReader directly: it prepares reads, leaves some unconsumed,
drains, and checks the registration count, that the abandoned rows were never written, and that the ring still
reads."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PREFIX = "SGLANG_EXPERT_STREAM_URING_"

_SOURCE = r'''
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe/expert_stream/host/uring_reader.h"

using sglang::expert_stream::FixedLeg;
using sglang::expert_stream::ReadCompletion;
using sglang::expert_stream::RegisteredRegion;
using sglang::expert_stream::UringReader;

constexpr size_t kRow = 65536, kRows = 64, kBytes = kRow * kRows;
constexpr unsigned kDepth = 16;
constexpr uint8_t kFill = 0xcd;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
uint8_t pattern(size_t i) { return static_cast<uint8_t>((i * 131 + (i >> 12) * 7 + 3) % 251); }

struct Fixture {
  uint8_t* slab = nullptr;
  std::string path;
  int fd = -1;
  bool direct = std::getenv("RINGRESET_DIRECT") != nullptr;
  explicit Fixture(const std::string& dir) {
    void* p = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    require(p != MAP_FAILED, "mmap failed");
    slab = static_cast<uint8_t*>(p);
    std::string pat = dir + "/ringreset-XXXXXX";
    std::vector<char> name(pat.begin(), pat.end());
    name.push_back('\0');
    const int w = mkstemp(name.data());
    require(w >= 0, "mkstemp failed");
    path = name.data();
    std::vector<uint8_t> bytes(kBytes);
    for (size_t i = 0; i < kBytes; ++i) bytes[i] = pattern(i);
    require(pwrite(w, bytes.data(), kBytes, 0) == static_cast<ssize_t>(kBytes) && fsync(w) == 0, "write failed");
    ::close(w);
    fd = ::open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    require(fd >= 0, "open failed");
  }
  ~Fixture() {
    if (fd >= 0) ::close(fd);
    if (!path.empty()) unlink(path.c_str());
    munmap(slab, kBytes);
  }
};

iovec g_iov[kRows];  // a readv SQE points at its iovec until the kernel consumes it

// Prepares a read of file row `row` into slab row `row`, tagged `row`.
void prep(UringReader& reader, const Fixture& fx, size_t row) {
  g_iov[row] = iovec{fx.slab + row * kRow, kRow};
  bool ok;
  if (reader.fixed_reads()) {
    FixedLeg leg[1];
    require(reader.fixed_legs(&g_iov[row], 1, leg) == 1, "a row must be one leg");
    ok = reader.prep_readv_fixed(fx.fd, &g_iov[row], 1, row * kRow, leg[0].buffer, row);
  } else {
    ok = reader.prep_readv(fx.fd, &g_iov[row], 1, row * kRow, row);
  }
  require(ok, "no SQE for row " + std::to_string(row));
}

bool untouched(const Fixture& fx, size_t row) {
  for (size_t i = 0; i < kRow; ++i)
    if (fx.slab[row * kRow + i] != kFill) return false;
  return true;
}

int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: harness <dir> unconsumed|mixed|refused|refused_fail");
    const std::string scenario = argv[2];
    Fixture fx(argv[1]);
    UringReader reader;
    require(reader.init(kDepth), "ring setup failed");
    if (reader.fixed_reads()) reader.set_fixed_chunk_cap(4 * kRow);  // 16 chunks: a real table, not one entry
    reader.configure_resources({fx.fd}, {RegisteredRegion{fx.slab, kBytes, kRow}}, fx.direct);
    std::memset(fx.slab, kFill, kBytes);

    // Rows 0-3 are prepared. `mixed` submits rows 0-1 first (consumed, in flight), then prepares 2-3 (unconsumed).
    size_t first_abandoned = 0;
    if (scenario == "mixed") {
      prep(reader, fx, 0);
      prep(reader, fx, 1);
      require(reader.submit(0) >= 0, "submit failed");
      first_abandoned = 2;
    }
    for (size_t row = first_abandoned; row < 4; ++row) prep(reader, fx, row);
    // (Task 2 inserts the refused / refused_fail hooks here.)

    const auto t0 = std::chrono::steady_clock::now();
    try {
      reader.drain(4);
    } catch (const std::runtime_error& e) {
      std::cout << "RAISED " << e.what() << "\nREADY " << reader.ready() << '\n';
      reader.close();
      std::cout << "CLOSED\n";
      return 0;
    }
    const double drain_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    for (size_t row = first_abandoned; row < 4; ++row)
      require(untouched(fx, row), "a discarded read wrote row " + std::to_string(row));

    // The ring still reads, and nothing from the drain surfaces: exactly one completion, the new read's.
    prep(reader, fx, 5);
    require(reader.submit(1) >= 0, "submit after drain failed");
    std::vector<ReadCompletion> out;
    while (out.empty()) {
      reader.reap(out);
      if (out.empty()) require(reader.submit(1) >= 0, "wait failed");
    }
    require(out.size() == 1 && out[0].data == 5, "a completion other than row 5's surfaced after the drain");
    require(out[0].res == static_cast<int>(kRow), "row 5 read " + std::to_string(out[0].res));
    for (size_t i = 0; i < kRow; ++i)
      require(fx.slab[5 * kRow + i] == pattern(5 * kRow + i), "row 5 bytes differ");

    std::cout << "REGISTRATIONS " << reader.registrations() << "\nSQ_SPACE " << reader.sq_space() << "\nDRAIN_MS "
              << std::fixed << std::setprecision(3) << drain_ms << "\nPASS\n";
    reader.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
'''


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    compiler = shutil.which("c++")
    assert compiler is not None, "the harness needs a C++ compiler"
    root = Path(__file__).resolve().parents[4]
    build = tmp_path_factory.mktemp("ringreset-build")
    source = build / "ringreset.cpp"
    source.write_text(_SOURCE)
    binary = build / "ringreset"
    built = subprocess.run(
        [compiler, "-std=c++20", "-O2", "-pthread", "-I", str(root / "python/sglang/kernels/jit/csrc"), str(source),
         "-luring", "-o", str(binary)],
        capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    return binary


def _run(harness, tmp_path, scenario, read_mode):
    env = {k: v for k, v in os.environ.items() if not k.startswith(PREFIX)}
    env[PREFIX + "FIXED_FILES"] = "1"  # the rewrite must also clear IOSQE_FIXED_FILE
    env[PREFIX + "READ_MODE"] = read_mode
    run = subprocess.run([str(harness), str(tmp_path), scenario], env=env, capture_output=True, text=True,
                         timeout=120, check=False)
    output = run.stdout + run.stderr
    if "unsupported by the running kernel" in output:
        pytest.skip(output)
    assert run.returncode == 0, output
    return run.stdout, output


def _value(stdout, key):
    match = re.search(rf"^{key} (\S+)$", stdout, re.M)
    assert match, stdout
    return match[1]


@pytest.mark.parametrize("scenario", ["unconsumed", "mixed"])
@pytest.mark.parametrize("read_mode", ["normal", "fixed", "readv_fixed"])
def test_discarding_unconsumed_sqes_keeps_the_ring_and_its_registrations(harness, tmp_path, scenario, read_mode):
    stdout, output = _run(harness, tmp_path, scenario, read_mode)
    assert "PASS" in stdout, output
    assert _value(stdout, "REGISTRATIONS") == "1", output  # a ring reset registers a second time
    assert _value(stdout, "SQ_SPACE") == "16", output
