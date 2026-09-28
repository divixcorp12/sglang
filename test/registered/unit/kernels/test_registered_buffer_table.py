"""Row-aligned registered-buffer chunks shared by UringFileReader and the expert-stream reader (plan
2026-09-28-reader-crtp-uring-registration Task 6). plan_chunks is address math (fake addresses, so tier-sized slabs cost
nothing); RegisteredBufferTable runs against a real ring on small buffers."""

import shutil
import subprocess
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SOURCE = r'''
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <sys/mman.h>
#include "io/registered_buffers.h"
using namespace sglang::io;
constexpr uint64_t G = 1ULL << 30;
template <class F> static bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

// Every row [base + k*row, +row) lies in exactly one chunk, and the chunks tile the slab in order.
static void every_row_in_one_chunk(uint64_t base, uint64_t bytes, uint64_t row, uint64_t cap) {
  const auto plan = plan_chunks(base, bytes, row, cap);
  uint64_t at = base;
  for (const auto& c : plan) {
    assert(c.base == at && c.length > 0 && c.length <= cap && c.length % row == 0);
    at += c.length;
  }
  assert(at == base + bytes);
  for (uint64_t k = 0; k < bytes / row; ++k) {
    const uint64_t lo = base + k * row, hi = lo + row;
    int holders = 0;
    for (const auto& c : plan) holders += c.base <= lo && hi <= c.base + c.length;
    assert(holders == 1);
  }
}

int main() {
  // A dsv41-sized named slab: 2.69 GB of 3,501,056 B rows (not a divisor of 1 GiB). Chunks are 306 rows
  // (1,071,323,136 B) and the last takes the remainder: the non-divisible last chunk.
  const uint64_t row = 3501056, rows = 768, base = 0x7f0000001000ull;
  auto plan = plan_chunks(base, rows * row, row, G);
  assert(plan.size() == 3);
  assert(plan[0].length == (G / row) * row && plan[1].length == plan[0].length);
  assert(plan[2].length == rows * row - 2 * plan[0].length && plan[2].length % row == 0);
  every_row_in_one_chunk(base, rows * row, row, G);
  // Divisible: 1 GiB of 4 KiB rows is exactly one chunk; one more row makes a 4 KiB last chunk.
  assert(plan_chunks(base, G, 4096, G).size() == 1);
  auto two = plan_chunks(base, G + 4096, 4096, G);
  assert(two.size() == 2 && two[1].length == 4096);
  // A lowered test cap over a small slab: 49152 B rows, 64 KiB cap -> one row per chunk.
  every_row_in_one_chunk(0x10000, 12 * 49152, 49152, 65536);
  assert(plan_chunks(0x10000, 12 * 49152, 49152, 65536).size() == 12);
  // row_bytes == 0 keeps UringFileReader's plain chunks from base.
  auto plain = plan_chunks(base, 2 * G + 5, 0, G);
  assert(plain.size() == 3 && plain[0].length == G && plain[2].length == 5);
  // Refusals: a row larger than the cap, a slab that is not whole rows.
  assert(throws([] { plan_chunks(0x1000, 4 * (G + 4096), G + 4096, G); }));
  assert(throws([] { plan_chunks(0x1000, 10000, 4096, G); }));

  // The table on a real ring: rows of 16 KiB, cap 64 KiB -> 4 rows per chunk; 10 rows -> 3 chunks (4, 4, 2).
  io_uring ring{};
  assert(io_uring_queue_init(8, &ring, 0) == 0);
  RegisteredBufferTable t;
  assert(t.init(&ring, 16));
  const size_t n = 10 * 16384;
  auto* mem = static_cast<uint8_t*>(mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  const uint64_t b = reinterpret_cast<uint64_t>(mem);
  assert(t.add(b, n, 16384, 65536) == 3 && t.chunks() == 3 && t.bytes() == n && t.largest() == 65536);
  for (uint64_t k = 0; k < 10; ++k) assert(t.find(b + k * 16384, 16384) >= 0);
  assert(t.find(b + 3 * 16384, 2 * 16384) == -1);  // rows 3 and 4 straddle chunks 0 and 1
  assert(t.add(b + 4096, 4096, 4096, 65536) == 0 && t.last_error_context() == "overlap");
  assert(t.remove_range(b, b + n) == 3 && t.chunks() == 0 && t.find(b, 16384) == -1);
  // No free slot: 16 slots, 17 chunks requested -> 0 and nothing left registered.
  auto* big = static_cast<uint8_t*>(mmap(nullptr, 17 * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  assert(t.add(reinterpret_cast<uint64_t>(big), 17 * 4096, 4096, 4096) == 0 && t.chunks() == 0);
  assert(t.last_error_context() == "no free slot");
  io_uring_queue_exit(&ring);
  std::puts("PASS registered buffer table");
  return 0;
}
'''


def test_registered_buffer_table(tmp_path):
    compiler = shutil.which("c++")
    assert compiler is not None
    root = Path(__file__).resolve().parents[4]
    source = tmp_path / "table.cpp"
    source.write_text(_SOURCE)
    binary = tmp_path / "table"
    built = subprocess.run(
        [compiler, "-std=c++20", "-O1", "-UNDEBUG", "-I", str(root / "python/sglang/kernels/jit/csrc"), str(source),
         "-luring", "-o", str(binary)], capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30, check=False)
    assert run.returncode == 0 and "PASS registered buffer table" in run.stdout, run.stdout + run.stderr
