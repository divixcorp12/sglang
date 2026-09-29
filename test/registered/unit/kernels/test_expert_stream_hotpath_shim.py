"""The counting shim (plan 2026-09-29-hotpath-zero-overhead Task 3), and the hot path's per-request allocator, mutex,
condvar, clock and sleep counts on the service and copy threads, measured on the build production loads."""

import subprocess
import textwrap

import pytest

from sglang.test import hotpath_shim
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

REQUESTS = 200


@pytest.fixture(scope="module")
def shim(tmp_path_factory):
    return hotpath_shim.build(tmp_path_factory.mktemp("shim"))


def test_the_shim_counts_a_named_threads_calls_and_nothing_else(shim, tmp_path):
    """Self-test of every kind, both roles and arming: while armed, a thread named x-ram-miss and one named x-copy-eng
    each make exactly 3 mallocs, 3 frees, 2 mutex locks, 1 condvar signal, 1 clock read, 1 (zero-length) sleep and
    1 futex wake through syscall(), and each lands in its own role; an unnamed thread's identical calls are not
    counted; and a third named thread (y-ram-miss) that makes the same calls after the shim is disarmed is recognized
    but adds nothing."""
    src = tmp_path / "probe.c"
    src.write_text(textwrap.dedent(r'''
        #define _GNU_SOURCE
        #include <pthread.h>
        #include <stdlib.h>
        #include <stdio.h>
        #include <time.h>
        #include <dlfcn.h>
        #include <linux/futex.h>
        #include <sys/syscall.h>
        #include <unistd.h>
        static int word;
        static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
        static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
        static void work(void) { struct timespec t, zero = {0, 0}; for (int i = 0; i < 3; ++i) free(malloc(64));
          pthread_mutex_lock(&m); pthread_mutex_unlock(&m); pthread_mutex_lock(&m); pthread_mutex_unlock(&m);
          pthread_cond_signal(&c); clock_gettime(CLOCK_MONOTONIC, &t); nanosleep(&zero, 0);
          syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, 1, 0, 0, 0); }
        static void* named(void* name) { pthread_setname_np(pthread_self(), (const char*)name); work(); return 0; }
        static void* plain(void* a) { work(); return 0; }
        static void run(void* (*fn)(void*), const char* name) { pthread_t t; pthread_create(&t, 0, fn, (void*)name);
          pthread_join(t, 0); }
        int main(void) { void (*arm)(int) = dlsym(RTLD_DEFAULT, "hotpath_shim_arm");
          long (*count)(int, int) = dlsym(RTLD_DEFAULT, "hotpath_shim_count");
          long (*seen)(int) = dlsym(RTLD_DEFAULT, "hotpath_shim_threads");
          arm(1); run(named, "x-ram-miss"); run(named, "x-copy-eng"); run(plain, 0);
          arm(0); run(named, "y-ram-miss");
          for (int th = 0; th < 2; ++th) { for (int k = 0; k < %d; ++k) printf("%%ld ", count(th, k)); }
          printf("%%ld %%ld\n", seen(0), seen(1)); return 0; }
    ''') % len(hotpath_shim.KINDS))
    exe = tmp_path / "probe"
    subprocess.run(["cc", "-O0", "-o", str(exe), str(src), "-ldl", "-lpthread"], check=True)
    out = subprocess.run([str(exe)], env={"LD_PRELOAD": str(shim)}, capture_output=True, text=True, check=True)
    values = list(map(int, out.stdout.split()))
    kinds = len(hotpath_shim.KINDS)
    counts = {th: dict(zip(hotpath_shim.KINDS, values[i * kinds:(i + 1) * kinds]))
              for i, th in enumerate(hotpath_shim.THREADS)}
    expected = {"malloc": 3, "free": 3, "mutex": 2, "cond": 1, "clock": 1, "sleep": 1, "futex": 1}
    assert counts == {"service": expected, "copy": expected}, counts
    # Two ram-miss threads were recognized (the disarmed one too), so its zero is "not counted", not "not seen".
    assert values[2 * kinds:] == [2, 1], values


def test_characterize_the_hot_path(shim, tmp_path):
    """Prints the per-request counts of the production build (the one a service loads with no trace or fault), so the
    baseline (master) and every later phase can be compared. It asserts that the measurement happened -- the
    requested window was measured, every kind was read for both threads, and the shim recognized exactly one service
    and one copy thread by name -- not what the counts are; the zero-count tests of later tasks do that."""
    counts = hotpath_shim.run_child(shim, variant="prod", requests=REQUESTS, tmp=tmp_path)
    assert counts["requests"] == REQUESTS
    for th in hotpath_shim.THREADS:
        assert set(counts[th]) == set(hotpath_shim.KINDS), (th, counts[th])
    # A shim that never recognized a thread reports all zeros; that must fail here, not pass as "zero overhead".
    assert counts["threads"] == {"service": 1, "copy": 1}, counts["threads"]
    per = {th: {k: round(v / counts["requests"], 2) for k, v in counts[th].items()} for th in hotpath_shim.THREADS}
    print("HOTPATH raw", counts)
    print("HOTPATH per request", per)


def test_the_prod_service_thread_reads_no_clock_per_request(shim, tmp_path):
    """Spec M3/M4/M8 and D6: the watchdog's episode word, turn-counted progress and iteration-budget pacing leave
    the service thread no clock read while it serves (the watchdog thread reads the clock instead)."""
    counts = hotpath_shim.run_child(shim, variant="prod", tmp=tmp_path)
    # A zero from a shim that never recognized the thread would prove nothing.
    assert counts["requests"] == REQUESTS and counts["threads"] == {"service": 1, "copy": 1}, counts
    assert counts["service"]["clock"] == 0, counts


@pytest.mark.parametrize("variant", ["prod", "instr"])
def test_the_service_thread_allocates_nothing_per_request(shim, tmp_path, variant):
    """Spec A1-A6, A9: after warm-up the service thread makes no allocator call while it serves (the instrumented
    build too, with its trace off: its metrics are fixed-size)."""
    counts = hotpath_shim.run_child(shim, variant=variant, tmp=tmp_path)
    # A zero from a shim that never recognized the thread would prove nothing.
    assert counts["requests"] == REQUESTS and counts["threads"] == {"service": 1, "copy": 1}, counts
    print("HOTPATH", variant, counts)
    assert counts["service"]["malloc"] == 0 and counts["service"]["free"] == 0, counts


def _measured(counts):
    """The window covered what the zeros below are about: every step served, copy jobs on the copy thread and a
    completed CopyDone for each, and a deferral (a request held back by a slot under a copy lease)."""
    assert counts["requests"] == REQUESTS and counts["threads"] == {"service": 1, "copy": 1}, counts
    assert counts["copy_jobs"] > 0 and counts["copies_done"] == counts["copy_jobs"], counts
    assert counts["deferrals"] >= 1, counts


@pytest.mark.parametrize("variant", ["prod", "instr"])
def test_the_copy_thread_allocates_nothing_and_waits_on_no_condvar(shim, tmp_path, variant):
    """Spec A7, A8, L8, L9 and 6.3 item 4: no per-pass deque, no queue node, no condition variable, a submit that nests
    no lock, and no lock at all on the copy thread: it hands each completed job back to the owner through an SPSC ring
    (Task 14), and the owner releases the COPYING lease. The service thread takes no lock either (Task 15: the tier
    has no mutex). With the tests' 200 us copy spin the copy thread goes to sleep between the
    Python-paced steps, so this run also covers the futex idle path: a wait on the copy thread each time it sleeps, and
    at most one wake per submitted job on the service thread (submit wakes only a thread that is asleep)."""
    counts = hotpath_shim.run_child(shim, variant=variant, tmp=tmp_path)
    print("HOTPATH", variant, counts)
    _measured(counts)
    service, copy = counts["service"], counts["copy"]
    assert service["malloc"] == 0 and service["free"] == 0, counts
    assert copy["malloc"] == 0 and copy["free"] == 0 and copy["cond"] == 0, counts
    assert service["cond"] == 0, counts
    assert copy["mutex"] == 0, counts
    assert service["mutex"] == 0, counts  # Task 15; before Task 14 a per-turn lock counted hundreds of thousands here
    assert copy["sleep"] == 0, counts  # the copy thread's idle wait is a futex, never a nanosleep
    if variant == "prod":
        assert copy["clock"] == 0, counts  # InstrBuild's copy latency and issue-time metrics read the clock
    assert service["futex"] <= counts["copy_jobs"], counts


def test_a_spinning_copy_thread_costs_the_service_no_syscall(shim, tmp_path):
    """Spec 6.3 item 5: while the copy thread spins (a 50 ms budget, which the window never exhausts) it never
    sleeps, so it makes no futex wait and the service's submit, which wakes only a sleeping thread, makes no futex
    wake: copy jobs cross from the service to the copy thread with no syscall at all."""
    counts = hotpath_shim.run_child(shim, variant="prod", tmp=tmp_path, copy_spin_us=50_000)
    print("HOTPATH spinning", counts)
    _measured(counts)
    assert counts["copy"]["futex"] == 0 and counts["service"]["futex"] == 0, counts
    assert counts["copy"]["cond"] == 0 and counts["copy"]["malloc"] == 0 and counts["service"]["malloc"] == 0, counts


@pytest.mark.parametrize("variant", ["prod", "instr"])
def test_the_service_and_copy_threads_take_no_lock_and_never_wait_on_a_condvar(shim, tmp_path, variant):
    """Spec L1-L10 and 6.3 (Task 15): over the measured requests -- copy jobs on the copy thread and deferrals among
    them -- neither thread takes a mutex, touches a condition variable or allocates, and the service thread never
    sleeps. The request path's only kernel wait is the io_uring completion (D5), which is not a lock, and the only
    futex calls are the copy engine's documented idle protocol: the copy thread's wait when it sleeps, and the service's
    wake in submit, at most one per job, only for a sleeping copy thread. The shim counts pthread mutexes and condvars
    only; that nothing else stands in for the tier mutex (a rwlock, a raw futex, a spin lock) is the source backstop's
    job (test_expert_stream_ownership.py::test_the_tier_declares_only_the_callers_mutex)."""
    counts = hotpath_shim.run_child(shim, variant=variant, tmp=tmp_path)
    print("HOTPATH no-lock", variant, counts)
    _measured(counts)
    for thread in ("service", "copy"):
        assert counts[thread]["mutex"] == 0 and counts[thread]["cond"] == 0, counts
        assert counts[thread]["malloc"] == 0 and counts[thread]["free"] == 0, counts
    assert counts["service"]["sleep"] == 0 and counts["copy"]["sleep"] == 0, counts
    assert counts["service"]["futex"] <= counts["copy_jobs"], counts
    if variant == "prod":
        assert counts["service"]["clock"] == 0 and counts["copy"]["clock"] == 0, counts
    else:
        # InstrBuild's metrics, and nothing else: copy_latency_ns's submit stamp on the service thread (one per job),
        # and on the copy thread copy_issue_ns's pair plus copy_latency_ns's completion stamp (three per job).
        assert counts["service"]["clock"] == counts["copy_jobs"], counts
        assert counts["copy"]["clock"] == 3 * counts["copy_jobs"], counts
