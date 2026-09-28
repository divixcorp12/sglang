"""Fixed buffers over a real slab larger than 1 GiB, on the real divix01 kernel (plan
2026-09-28-reader-crtp-uring-registration, Task 8).

A native harness maps a populated 1.5 GiB slab of dsv41 rows (460 x 3,501,056 B) and a 64-row slab of 49,152 B rows,
configures the production UringReader from the environment (READ_MODE fixed or readv_fixed, FIXED_FILES=1) with both
slabs as RegisteredRegions under the default 1 GiB cap, and reads a 16 MiB O_DIRECT file into them through
fixed_legs + prep_readv_fixed. It proves that the big slab registers as 306 + 154 rows (chunk 1 starts at
306 x 3,501,056), that the two rows around the cut land in buffers 0 and 1, that an iovec straddling the cut is
refused, and that a two-slab destination fans out into two legs submitted together.

A second case lowers RLIMIT_MEMLOCK to 256 KiB in the child: the ring still sets up, and the real registration
ENOMEM must surface as the explicit fixed-buffer refusal naming the limit.

Run on divix01 (see the brief's Step 2 command): under rowimg-disk.lock, numactl --membind=1, taskset -c 0-63.
"""

import os
import re
import resource
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.uname().nodename != "divix01", reason="1.5 GiB pinned: divix01 only")

_SOURCE = r'''
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe/expert_stream/host/uring_reader.h"

using sglang::expert_stream::FixedLeg;
using sglang::expert_stream::ReadCompletion;
using sglang::expert_stream::RegisteredRegion;
using sglang::expert_stream::UringReader;

constexpr size_t kBigRow = 3501056, kBigRows = 460, kBigBytes = kBigRow * kBigRows;  // 1,610,485,760 B
constexpr size_t kSmallRow = 49152, kSmallRows = 64, kSmallBytes = kSmallRow * kSmallRows;
constexpr size_t kFileBytes = 16u << 20;
constexpr size_t kCut = 306;  // floor(1 GiB / 3,501,056): rows in chunk 0

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
std::string setting(const char* key, const char* fallback) {
  const char* value = std::getenv(key);
  return value ? value : fallback;
}

struct Slab {
  uint8_t* data;
  size_t bytes;
  explicit Slab(size_t n) : bytes(n) {
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    require(p != MAP_FAILED, "mmap slab failed: " + std::string(std::strerror(errno)));
    data = static_cast<uint8_t*>(p);
  }
  ~Slab() { munmap(data, bytes); }
};

struct DataFile {
  std::string path;
  std::vector<uint8_t> bytes;
  int fd = -1;
  explicit DataFile(const std::string& directory) : bytes(kFileBytes) {
    std::string pattern = directory + "/fixed-big-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int writer = mkstemp(name.data());
    require(writer >= 0, "mkstemp failed");
    path = name.data();
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>((i * 131 + (i >> 12) * 7 + 3) % 251);
    const ssize_t written = pwrite(writer, bytes.data(), bytes.size(), 0);
    const int synced = fsync(writer);
    ::close(writer);
    require(written == static_cast<ssize_t>(bytes.size()) && synced == 0, "fixture write/fsync failed");
    fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    require(fd >= 0, "opening fixture O_DIRECT failed: " + std::string(std::strerror(errno)));
  }
  ~DataFile() {
    if (fd >= 0) ::close(fd);
    if (!path.empty()) unlink(path.c_str());
  }
};

// Prepares every leg of one read as its own SQE at its file offset; returns the SQE count.
unsigned prep_legs(UringReader& reader, int fd, const iovec* iov, unsigned count, uint64_t off, uint64_t tag,
                   std::vector<FixedLeg>& legs_out) {
  FixedLeg legs[8];
  const unsigned n = reader.fixed_legs(iov, count, legs);
  legs_out.assign(legs, legs + n);
  reader.note_fanout(n);
  uint64_t at = off;
  for (unsigned l = 0; l < n; ++l) {
    require(reader.prep_readv_fixed(fd, iov + legs[l].first, legs[l].count, at, legs[l].buffer, tag + l),
            "no SQE for leg " + std::to_string(l));
    at += legs[l].bytes;
  }
  return n;
}

std::vector<ReadCompletion> settle(UringReader& reader, unsigned n) {
  const int submitted = reader.submit(n);
  require(submitted >= 0, "submit failed: " + std::to_string(submitted));
  std::vector<ReadCompletion> out;
  while (out.size() < n) {
    reader.reap(out);
    if (out.size() < n) require(reader.submit(1) >= 0, "wait failed");
  }
  require(out.size() == n, "unexpected extra completion");
  return out;
}

void expect(const std::vector<ReadCompletion>& done, uint64_t tag, size_t bytes) {
  for (const auto& c : done)
    if (c.data == tag) {
      require(c.res == static_cast<int>(bytes), "tag " + std::to_string(tag) + " read " + std::to_string(c.res));
      return;
    }
  throw std::runtime_error("completion for tag " + std::to_string(tag) + " lost");
}

void same(const uint8_t* got, const DataFile& file, size_t off, size_t n, const char* what) {
  require(std::memcmp(got, file.bytes.data() + off, n) == 0, std::string("bytes differ: ") + what);
}

int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: harness <dir> read|memlock");
    const std::string read_mode = setting("SGLANG_EXPERT_STREAM_URING_READ_MODE", "");
    require(read_mode == "fixed" || read_mode == "readv_fixed", "READ_MODE must be fixed or readv_fixed");
    require(setting("SGLANG_EXPERT_STREAM_URING_FIXED_FILES", "0") == "1", "FIXED_FILES must be 1");
    Slab big(kBigBytes), small(kSmallBytes);
    DataFile file(argv[1]);
    const std::vector<int> fds{file.fd};
    const std::vector<RegisteredRegion> regions{{big.data, kBigBytes, kBigRow}, {small.data, kSmallBytes, kSmallRow}};

    // The chunk list the table registers (plan_chunks is what register_resources feeds it). Rows must not straddle.
    size_t chunks = 0;
    for (const auto& r : regions) {
      for (const auto& c : sglang::io::plan_chunks(reinterpret_cast<uint64_t>(r.base), r.bytes, r.row_bytes,
                                                   sglang::io::kMaxRegisteredBufferBytes)) {
        const uint64_t start = c.base - reinterpret_cast<uint64_t>(r.base);
        require(start % r.row_bytes == 0 && c.length % r.row_bytes == 0, "chunk boundary off a row boundary");
        std::cout << "CHUNK region_row=" << r.row_bytes << " start=" << start << " len=" << c.length
                  << " rows=" << c.length / r.row_bytes << '\n';
        ++chunks;
      }
    }

    UringReader reader;
    require(reader.init(8), "ring setup failed");
    if (std::string(argv[2]) == "memlock") {
      try {
        reader.configure_resources(fds, regions, true);
      } catch (const std::exception& e) {
        std::cout << "REFUSED " << e.what() << '\n';
        return 0;
      }
      throw std::runtime_error("registration under a 256 KiB memlock limit was accepted");
    }
    const auto t0 = std::chrono::steady_clock::now();
    reader.configure_resources(fds, regions, true);
    const double register_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    require(chunks == 3, "expected 3 chunks (306 + 154 rows, then the small slab), planned " + std::to_string(chunks));

    // Every row lies in one buffer: big rows [0, 306) in buffer 0, [306, 460) in 1, small rows in 2.
    for (size_t k = 0; k < kBigRows; ++k) {
      iovec v{big.data + k * kBigRow, kBigRow};
      FixedLeg leg[1];
      require(reader.fixed_legs(&v, 1, leg) == 1 && leg[0].buffer == (k < kCut ? 0 : 1),
              "big row " + std::to_string(k) + " in the wrong buffer");
    }
    for (size_t k = 0; k < kSmallRows; ++k) {
      iovec v{small.data + k * kSmallRow, kSmallRow};
      FixedLeg leg[1];
      require(reader.fixed_legs(&v, 1, leg) == 1 && leg[0].buffer == 2, "small row in the wrong buffer");
    }
    // A range across the cut lies in no registered buffer: chunk 1 starts exactly at 306 x 3,501,056.
    {
      iovec across{big.data + kCut * kBigRow - 4096, 8192};
      FixedLeg leg[1];
      bool refused = false;
      try {
        reader.fixed_legs(&across, 1, leg);
      } catch (const std::invalid_argument&) {
        refused = true;
      }
      require(refused, "an iovec straddling the chunk cut was accepted");
    }

    std::memset(big.data, 0xcd, kBigBytes);
    std::memset(small.data, 0xcd, kSmallBytes);
    std::vector<FixedLeg> legs;

    // Row 305 (last of chunk 0) and row 306 (first of chunk 1): one leg each, buffers 0 and 1, one SQE each.
    const iovec row305{big.data + 305 * kBigRow, kBigRow}, row306{big.data + 306 * kBigRow, kBigRow};
    require(prep_legs(reader, file.fd, &row305, 1, 0, 100, legs) == 1 && legs[0].buffer == 0, "row 305 leg");
    require(prep_legs(reader, file.fd, &row306, 1, kBigRow, 200, legs) == 1 && legs[0].buffer == 1, "row 306 leg");
    auto done = settle(reader, 2);
    expect(done, 100, kBigRow);
    expect(done, 200, kBigRow);
    same(big.data + 305 * kBigRow, file, 0, kBigRow, "row 305");
    same(big.data + 306 * kBigRow, file, kBigRow, kBigRow, "row 306");

    // One two-iovec destination across slabs (big row 459, small row 7) at consecutive file offsets: two legs,
    // prepared at their offsets and submitted together.
    const uint64_t off = 2 * kBigRow;
    const iovec pair[]{{big.data + 459 * kBigRow, kBigRow}, {small.data + 7 * kSmallRow, kSmallRow}};
    require(prep_legs(reader, file.fd, pair, 2, off, 300, legs) == 2, "a two-slab read must be 2 legs");
    require(legs[0].buffer == 1 && legs[1].buffer == 2, "two-slab legs in the wrong buffers");
    done = settle(reader, 2);
    expect(done, 300, kBigRow);
    expect(done, 301, kSmallRow);
    same(big.data + 459 * kBigRow, file, off, kBigRow, "big row 459");
    same(small.data + 7 * kSmallRow, file, off + kBigRow, kSmallRow, "small row 7");
    require(small.data[6 * kSmallRow + kSmallRow - 1] == 0xcd && small.data[8 * kSmallRow] == 0xcd,
            "fan-out wrote outside its rows");
    require(reader.fixed_cuts() == 1 && reader.fanout_sqes() == 2, "fan-out counters");
    reader.close();
    std::cout << "PASS big fixed slab chunks=" << chunks << " register_ms=" << register_ms
              << " read_mode=" << read_mode << '\n';
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
    root = Path(__file__).resolve().parents[3]
    build = tmp_path_factory.mktemp("fixed-big-build")
    source = build / "fixed_big.cpp"
    source.write_text(_SOURCE)
    binary = build / "fixed_big"
    built = subprocess.run(
        [compiler, "-std=c++20", "-O2", "-pthread", "-I", str(root / "python/sglang/kernels/jit/csrc"), str(source),
         "-luring", "-o", str(binary)],
        capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    return binary


def _env():
    env = dict(os.environ)
    env.setdefault("SGLANG_EXPERT_STREAM_URING_READ_MODE", "readv_fixed")
    env.setdefault("SGLANG_EXPERT_STREAM_URING_FIXED_FILES", "1")
    env["SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS"] = "1"
    return env


def test_big_fixed_slab_real_kernel(harness, tmp_path):
    run = subprocess.run([str(harness), str(tmp_path), "read"], env=_env(), capture_output=True, text=True,
                         timeout=300, check=False)
    output = run.stdout + run.stderr
    print(output)
    assert run.returncode == 0, output
    assert "PASS big fixed slab chunks=3 " in run.stdout, output
    chunks = re.findall(r"CHUNK region_row=(\d+) start=(\d+) len=(\d+)", run.stdout)
    assert [(int(r), int(s), int(n)) for r, s, n in chunks] == [
        (3501056, 0, 306 * 3501056), (3501056, 306 * 3501056, 154 * 3501056), (49152, 0, 64 * 49152)], output
    # The reader's own diagnostics agree: 3 chunks registered, the largest the 306-row one.
    diag = re.search(r"chunks=(\d+) largest_chunk=(\d+) chunk_cap=\d+ register_ms=([\d.]+)", run.stderr)
    assert diag and int(diag[1]) == 3 and int(diag[2]) == 306 * 3501056, output
    print(f"DIAG chunks={diag[1]} register_ms={diag[3]}")


def test_big_fixed_slab_memlock_refused(harness, tmp_path):
    limit = 262144

    def lower_memlock():
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (limit, limit))

    run = subprocess.run([str(harness), str(tmp_path), "memlock"], env=_env(), capture_output=True, text=True,
                         timeout=300, check=False, preexec_fn=lower_memlock)
    output = run.stdout + run.stderr
    print(output)
    assert run.returncode == 0, output
    # The registration refusal, not the ring-setup one, naming the limit and the kernel's ENOMEM.
    assert "REFUSED expert stream registering fixed buffers" in run.stdout, output
    assert f"RLIMIT_MEMLOCK={limit}" in run.stdout and "Cannot allocate memory" in run.stdout, output
