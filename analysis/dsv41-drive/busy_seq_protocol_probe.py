"""Direct observations of the kBusySeq page word, from a poller against the real service thread.

What it establishes (CPU, the host service over a simulated device page; no GPU):
  1. During a request's service the word equals the request's sequence, and it is observed many times
     (an injected 50 ms read makes the window wide enough for a Python poller).
  2. The word is not seen equal to seq once demand_done has reached seq. This is a SANITY check, not a
     test of the order: a poller cannot land in the sub-microsecond gap, so a service that cleared the word
     AFTER the done store still passes (busy_seq_mutants.sh, mutant busy_cleared_after_done_store, 3 of 3
     runs). That "the word clears before demand_done" holds is read from handle_demand and pump_demand
     (one thread; the clear is a release store in handle_demand, the done store is a release store after
     it returns); it is not established by execution. Between the clear and the done store the word is 0
     with done pending: reported as gap_polls, informational.
  3. An advisory in service does NOT set the word: it stays 0 throughout.
What it does NOT establish: how often a DEVICE poll lands inside the window. That needs a kernel under
the GPU lock (poll both words from launch, log first-seen latency against a host stamp of the store).

Run: python busy_seq_protocol_probe.py     (exit 1 on a failed property)
The same checks are pytest tests here (test_*): python -m pytest busy_seq_protocol_probe.py
"""

import pathlib
import tempfile
import time

import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup


def _host(capacity=6):
    s = ram_miss_setup(pathlib.Path(tempfile.mkdtemp(prefix="busy_seq_probe_")), capacity=capacity, experts=6)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    return page, host


def demand_window(delay_s=0.05, poll_s=0.4):
    """Poll busy_seq and demand_done through one demand whose read sleeps ``delay_s`` inside serve()."""
    page, host = _host()
    host.start_thread(fatal_wait_s=5.0)
    try:
        host.inject(delay_s=delay_s, delay_after_demands=0)
        seq = sim_post(page, 1, need=[2], protect=[2])
        polls = busy_polls = order_violations = gap_polls = 0
        opened = False
        end = time.perf_counter() + poll_s
        while time.perf_counter() < end:
            done, busy = page_word(page, "demand_done"), page_word(page, "busy_seq")
            polls += 1
            if busy == seq:
                busy_polls += 1
                opened = True
                order_violations += done >= seq
            elif opened and busy == 0 and done < seq:
                gap_polls += 1
            if opened and busy == 0 and done >= seq and time.perf_counter() > end - poll_s * 0.75:
                break
        return dict(seq=seq, polls=polls, busy_polls=busy_polls, order_violations=order_violations,
                    gap_polls=gap_polls, final_busy=page_word(page, "busy_seq"), final_done=page_word(page, "demand_done"))
    finally:
        host.stop()


def advisory_window(delay_s=0.1, poll_s=0.08):
    page, host = _host()
    host.start_thread(fatal_wait_s=5.0)
    try:
        host.inject(delay_s=delay_s, delay_after_demands=10**6)  # only advisories sleep
        sim_post(page, 1, need=[3], protect=[3], advisory=True, after=page_word(page, "demand_head") + 10)
        seen, in_service = set(), False
        end = time.perf_counter() + poll_s
        while time.perf_counter() < end:
            seen.add(page_word(page, "busy_seq"))
            in_service = in_service or host.counters()["advisories"] > 0
        return dict(in_service=in_service, busy_values=sorted(seen))
    finally:
        host.stop()


def test_the_word_equals_seq_during_service_and_is_clear_after_it():
    r = demand_window()
    assert r["busy_polls"] > 100, r  # the window is wide (an injected 50 ms read) and is seen many times
    assert r["order_violations"] == 0, r  # sanity only: cannot catch a clear placed after the done store
    assert r["final_busy"] == 0 and r["final_done"] == r["seq"], r


def test_an_advisory_in_service_leaves_the_word_at_zero():
    r = advisory_window()
    assert r["in_service"], r  # the advisory really was being served during the polling
    assert r["busy_values"] == [0], r


if __name__ == "__main__":
    demand = demand_window()
    advisory = advisory_window()
    print("demand :", demand)
    print("advisory:", advisory)
    ok = (
        demand["busy_polls"] > 100
        and demand["order_violations"] == 0
        and demand["final_busy"] == 0
        and demand["final_done"] == demand["seq"]
        and advisory["in_service"]
        and advisory["busy_values"] == [0]
    )
    print("properties hold" if ok else "PROPERTY FAILED")
    raise SystemExit(0 if ok else 1)
