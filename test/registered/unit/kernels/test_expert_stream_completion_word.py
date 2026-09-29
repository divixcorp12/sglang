"""The copy engine's completion word (phase 2 Task P2, results.md section 9c): CompletionWord's wrap-safe compare, its
clockless liveness check, the lost-write rule's re-load after an idle stream (the P1 review's C1 race: a word that
lands between the poll's load and the stream query must not fail stop), and the acquire that orders what the word
guards (under ThreadSanitizer where the compiler has it: a relaxed load is a reported race there)."""

import subprocess

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

HEADER = '#include "{moe}/expert_stream/host/completion_word.h"\n'

RULES = HEADER + '''
#include <cassert>
#include <cstdint>
using namespace sglang::expert_stream;
constexpr int kDone = CompletionWord::kDone;
constexpr int kPending = CompletionWord::kPending;
constexpr int kBusy = CompletionWord::kNotReady;
int main() {{
  alignas(64) uint32_t word = 0;
  // Wrap-safe: (int32_t)(word - token) >= 0, correct across the 2^32 wrap while fewer than 2^31 jobs are outstanding.
  assert(CompletionWord::reached(5, 5) && CompletionWord::reached(6, 5) && !CompletionWord::reached(4, 5));
  assert(CompletionWord::reached(2u, 0xFFFFFFFEu) && !CompletionWord::reached(0xFFFFFFFEu, 2u));
  assert(CompletionWord::reached(0u, 0xFFFFFFFFu) && !CompletionWord::reached(0xFFFFFFFFu, 0u));
  {{  // A pending head asks the stream once per `check_every` pending polls, never in between; busy keeps it pending.
    CompletionWord w(4);
    w.bind(&word);
    int asks = 0;
    auto busy = [&] {{ ++asks; return kBusy; }};
    for (int i = 0; i < 12; ++i) assert(w.poll(1, busy) == kPending);
    assert(asks == 3);
    word = 1;
    assert(w.poll(1, busy) == kDone && asks == 3);  // a reached word asks nothing
    assert(w.poll(0, busy) == kDone && asks == 3);  // an older token is done too (stream order)
    // A completion restarts the budget: the next head asks only after `check_every` more pending polls.
    for (int i = 0; i < 3; ++i) assert(w.poll(2, busy) == kPending);
    assert(asks == 3 && w.poll(2, busy) == kPending && asks == 4);
  }}
  {{  // A sticky stream error is returned as is: the engine fails stop (E5, the leases stay held).
    word = 0;
    CompletionWord w(2);
    w.bind(&word);
    auto faulted = [] {{ return 719; }};  // CUDA_ERROR_LAUNCH_FAILED
    assert(w.poll(1, faulted) == kPending && w.poll(1, faulted) == 719);
  }}
  {{  // An idle stream with the word still short after the re-load: a lost write, which fails stop.
    word = 0;
    CompletionWord w(1);
    w.bind(&word);
    auto idle = [] {{ return 0; }};
    assert(w.poll(1, idle) == CompletionWord::kLostWrite);
  }}
  {{  // The C1 interleaving, forced: the word lands after the poll's load and before the query reports the stream idle.
    // The re-load after SUCCESS sees it, so the engine completes the job and does not fail stop.
    word = 1;
    CompletionWord w(1);
    w.bind(&word);
    int asks = 0;
    auto lands_then_idle = [&] {{ ++asks; __atomic_store_n(&word, 2u, __ATOMIC_RELEASE); return 0; }};
    assert(w.poll(2, lands_then_idle) == kDone && asks == 1);
  }}
  {{  // The same across the wrap.
    word = 0xFFFFFFFFu;
    CompletionWord w(1);
    w.bind(&word);
    auto lands_then_idle = [&] {{ __atomic_store_n(&word, 0u, __ATOMIC_RELEASE); return 0; }};
    assert(w.poll(0u, lands_then_idle) == kDone);
  }}
  return 0;
}}
'''

# The GPU's analog on a CPU thread: payload writes, then the word (release), as the copy stream's DMA precedes its
# fenced word write. The consumer polls through CompletionWord and then reads the payload: only the poll's acquire
# orders those reads after the writes, so a relaxed load is a data race TSan reports.
ORDER = HEADER + '''
#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
using namespace sglang::expert_stream;
int main() {{
  alignas(64) uint32_t word = 0;
  uint64_t payload[8] = {{}};
  std::atomic<uint32_t> taken{{0}};
  constexpr uint32_t kRounds = {rounds};
  std::thread gpu([&] {{
    for (uint32_t r = 1; r <= kRounds; ++r) {{
      while (taken.load(std::memory_order_acquire) != r - 1) {{}}
      for (uint64_t& p : payload) p = r;
      __atomic_store_n(&word, r, __ATOMIC_RELEASE);
    }}
  }});
  CompletionWord w;
  w.bind(&word);
  auto busy = [] {{ return CompletionWord::kNotReady; }};
  for (uint32_t r = 1; r <= kRounds; ++r) {{
    while (w.poll(r, busy) != CompletionWord::kDone) {{}}
    for (const uint64_t p : payload) assert(p == r);
    taken.store(r, std::memory_order_release);
  }}
  gpu.join();
  return 0;
}}
'''


def _build_and_run(tmp_path, name, program, sanitize):
    src = tmp_path / f"{name}.cpp"
    src.write_text(program)
    exe = tmp_path / name
    flags = ["-fsanitize=thread", "-O1", "-g"] if sanitize else ["-O2"]
    built = subprocess.run(["c++", "-std=c++20", *flags, "-o", str(exe), str(src), "-lpthread"],
                           capture_output=True, text=True)
    if sanitize and built.returncode != 0:
        pytest.skip(f"no ThreadSanitizer here: {built.stderr[-300:]}")
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, env={"TSAN_OPTIONS": "halt_on_error=1"})
    assert run.returncode == 0, run.stderr[-3000:]


def test_the_completion_word_rules_hold(tmp_path):
    """Wrap-safe compare, one stream query per budget, error and lost-write verdicts, and the forced C1 interleaving."""
    _build_and_run(tmp_path, "rules", RULES.format(moe=MOE), sanitize=False)


@pytest.mark.parametrize("sanitize", [False, True], ids=["plain", "tsan"])
def test_the_poll_orders_what_the_word_guards(tmp_path, sanitize):
    _build_and_run(tmp_path, "order", ORDER.format(moe=MOE, rounds=20_000 if sanitize else 1_000_000), sanitize)
