"""SpscRing and FixedDeque (plan 2026-09-29-hotpath-zero-overhead Task 12): order, completeness, full and empty, under
a real producer thread (and ThreadSanitizer where the compiler has it)."""

import subprocess

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PROGRAM = '''
#include "{moe}/expert_stream/host/spsc_ring.h"
#include <cassert>
#include <cstdint>
#include <thread>
using namespace sglang::expert_stream;
int main() {{
  SpscRing<uint64_t, 32> ring;
  uint64_t v = 0;
  assert(ring.empty() && !ring.pop(&v) && ring.front() == nullptr);
  for (uint64_t i = 0; i < 32; ++i) assert(ring.push(i));
  assert(!ring.push(99));  // full at exactly N
  for (uint64_t i = 0; i < 32; ++i) {{ assert(ring.front() && *ring.front() == i && ring.pop(&v) && v == i); }}
  assert(ring.front() == nullptr);  // front() peeks: it never removes (Task 13's snapshot-only drain)
  constexpr uint64_t kItems = {items};
  std::thread producer([&] {{ for (uint64_t i = 1; i <= kItems; ++i) while (!ring.push(i)) {{}} }});
  uint64_t next = 1;
  while (next <= kItems) if (ring.pop(&v)) {{ assert(v == next); ++next; }}
  producer.join();
  FixedDeque<int, 8> d; d.push_back(1); d.push_back(2); d.push_back(3);
  d.erase_if([](int x) {{ return x == 2; }});
  assert(d.size() == 2 && d.front() == 1 && d[1] == 3); d.pop_front(); assert(d.front() == 3);
  return 0;
}}
'''


@pytest.mark.parametrize("sanitize", [False, True], ids=["plain", "tsan"])
def test_the_ring_delivers_every_item_in_order_across_threads(tmp_path, sanitize):
    src = tmp_path / "ring.cpp"
    src.write_text(PROGRAM.format(moe=MOE, items=200_000 if sanitize else 10_000_000))
    exe = tmp_path / "ring"
    flags = ["-fsanitize=thread", "-O1", "-g"] if sanitize else ["-O2"]
    built = subprocess.run(["c++", "-std=c++20", *flags, "-o", str(exe), str(src), "-lpthread"],
                           capture_output=True, text=True)
    if sanitize and built.returncode != 0:
        pytest.skip(f"no ThreadSanitizer here: {built.stderr[-300:]}")
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, env={"TSAN_OPTIONS": "halt_on_error=1"})
    assert run.returncode == 0, run.stderr[-3000:]
