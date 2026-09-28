"""Exercise the production completion thread with native atomic publications on CPU."""

import shutil
import subprocess
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_completion_bridge_timeout_abort_and_reuse(tmp_path: Path):
    compiler = shutil.which("c++")
    assert compiler is not None, "the CPU kernel tests require a C++ compiler"
    root = Path(__file__).resolve().parents[4]
    source = tmp_path / "completion.cpp"
    source.write_text(
        r'''#include <cassert>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include "moe/expert_stream/host/wait_completion.h"

using namespace sglang::expert_stream;
using namespace sglang::expert_stream::wire;
using namespace std::chrono_literals;

alignas(64) uint8_t page[kPageBytes]{};
alignas(64) uint8_t mailbox[kWaitCompletionBytes]{};
alignas(64) uint8_t lease[128]{};

void put32(uint8_t* p, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(p), value, __ATOMIC_RELEASE);
}
void put64(uint8_t* p, uint64_t value) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(p), value, __ATOMIC_RELEASE);
}
uint32_t get32(uint8_t* p) {
  return __atomic_load_n(reinterpret_cast<uint32_t*>(p), __ATOMIC_ACQUIRE);
}
uint64_t get64(uint8_t* p) {
  return __atomic_load_n(reinterpret_cast<uint64_t*>(p), __ATOMIC_ACQUIRE);
}
void prepare(uint64_t generation, uint64_t duration = 2000000000ull) {
  put32(mailbox + kWaitCompletionReady, 0);
  put64(mailbox + kWaitCompletionTimeoutNs, duration);
  put64(mailbox + kWaitCompletionToken, generation);
}
uint64_t wait_tag(uint64_t generation) {
  const auto limit = std::chrono::steady_clock::now() + 3s;
  while (get32(mailbox + kWaitCompletionReady) != 1) {
    assert(std::chrono::steady_clock::now() < limit);
    std::this_thread::sleep_for(50us);
  }
  const uint64_t token = get64(mailbox + kWaitCompletionToken);
  assert((token & ((1ull << 56) - 1)) == generation);
  return token >> 56;
}
template<class F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}

int main() {
  const int64_t handle = WaitCompletionRegistry::open(page, mailbox, lease);
  rejects([&] { WaitCompletionRegistry::open(page, mailbox, lease); });
  rejects([&] { WaitCompletionRegistry::close(handle + 12345); });
  // An idle/zero mailbox is not a request.
  std::this_thread::sleep_for(2ms);
  assert(get32(mailbox + kWaitCompletionReady) == 0);
  // No early wake; acquiring ready also exposes the final tagged token.
  prepare(1);
  std::this_thread::sleep_for(2ms);
  assert(get32(mailbox + kWaitCompletionReady) == 0);
  put32(page + kDemandDone, 1);
  assert(wait_tag(1) == kWaitTagReady);
  // Reuse must not inherit either the old ready flag or the old demand_done.
  prepare(2);
  std::this_thread::sleep_for(2ms);
  assert(get32(mailbox + kWaitCompletionReady) == 0);
  put32(page + kDemandDone, 2);
  assert(wait_tag(2) == kWaitTagReady);
  // A request that never reaches service times out independently of busy/read state.
  prepare(3, 2000000);
  assert(wait_tag(3) == kWaitTagTimeout);
  // A late completion for the timed-out generation must not wake the next one.
  prepare(4);
  put32(page + kDemandDone, 3);
  std::this_thread::sleep_for(2ms);
  assert(get32(mailbox + kWaitCompletionReady) == 0);
  put32(page + kFatal, 1);
  assert(wait_tag(4) == kWaitTagAborted);
  put32(page + kFatal, 0);
  prepare(5);
  put32(lease + kLeaseHeaderShutdown, 1);
  assert(wait_tag(5) == kWaitTagAborted);
  put32(lease + kLeaseHeaderShutdown, 0);
  // An immediate bypass published by preparation is left untouched by the bridge.
  put64(mailbox + kWaitCompletionToken, (uint64_t(kWaitTagBypass) << 56) | 6);
  put32(mailbox + kWaitCompletionReady, 1);
  std::this_thread::sleep_for(2ms);
  assert(wait_tag(6) == kWaitTagBypass);
  // Close aborts a pending request, joins its publisher, and releases registry ownership.
  prepare(7);
  WaitCompletionRegistry::close(handle);
  assert(wait_tag(7) == kWaitTagAborted);
  rejects([&] { WaitCompletionRegistry::close(handle); });
  const int64_t reopened = WaitCompletionRegistry::open(page, mailbox, nullptr);
  // seq32 rollover uses modular done comparison; epoch remains in the token.
  put32(page + kDemandDone, 0xffffffffu);
  prepare((1ull << 32) | 1);
  std::this_thread::sleep_for(2ms);
  assert(get32(mailbox + kWaitCompletionReady) == 0);
  put32(page + kDemandDone, 1);
  assert(wait_tag((1ull << 32) | 1) == kWaitTagReady);
  // Rapid reuse exercises the boundary after the publisher's ready store.
  for (uint32_t seq = 2; seq <= 100; ++seq) {
    const uint64_t generation = (1ull << 32) | seq;
    prepare(generation);
    assert(get32(mailbox + kWaitCompletionReady) == 0);
    put32(page + kDemandDone, seq);
    assert(wait_tag(generation) == kWaitTagReady);
  }
  // Cancellation keeps the publisher alive while queued GPU preparations drain.
  WaitCompletionRegistry::cancel(reopened);
  for (uint32_t seq = 101; seq <= 103; ++seq) {
    const uint64_t generation = (1ull << 32) | seq;
    prepare(generation);
    assert(wait_tag(generation) == kWaitTagAborted);
  }
  WaitCompletionRegistry::close(reopened);
  rejects([&] { WaitCompletionRegistry::cancel(reopened); });
  return 0;
}
'''
    )
    binary = tmp_path / "completion"
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-pthread", "-I", str(root / "python/sglang/kernels/jit/csrc"), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=15, capture_output=True, text=True)
