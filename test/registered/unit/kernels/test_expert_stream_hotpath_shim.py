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
    each make exactly 3 mallocs, 3 frees, 2 mutex locks, 1 condvar signal, 1 clock read and 1 (zero-length) sleep,
    and each lands in its own role; an unnamed thread's identical calls are not counted; and a third named thread
    (y-ram-miss) that makes the same calls after the shim is disarmed is recognized but adds nothing."""
    src = tmp_path / "probe.c"
    src.write_text(textwrap.dedent(r'''
        #define _GNU_SOURCE
        #include <pthread.h>
        #include <stdlib.h>
        #include <stdio.h>
        #include <time.h>
        #include <dlfcn.h>
        static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
        static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
        static void work(void) { struct timespec t, zero = {0, 0}; for (int i = 0; i < 3; ++i) free(malloc(64));
          pthread_mutex_lock(&m); pthread_mutex_unlock(&m); pthread_mutex_lock(&m); pthread_mutex_unlock(&m);
          pthread_cond_signal(&c); clock_gettime(CLOCK_MONOTONIC, &t); nanosleep(&zero, 0); }
        static void* named(void* name) { pthread_setname_np(pthread_self(), (const char*)name); work(); return 0; }
        static void* plain(void* a) { work(); return 0; }
        static void run(void* (*fn)(void*), const char* name) { pthread_t t; pthread_create(&t, 0, fn, (void*)name);
          pthread_join(t, 0); }
        int main(void) { void (*arm)(int) = dlsym(RTLD_DEFAULT, "hotpath_shim_arm");
          long (*count)(int, int) = dlsym(RTLD_DEFAULT, "hotpath_shim_count");
          long (*seen)(int) = dlsym(RTLD_DEFAULT, "hotpath_shim_threads");
          arm(1); run(named, "x-ram-miss"); run(named, "x-copy-eng"); run(plain, 0);
          arm(0); run(named, "y-ram-miss");
          for (int th = 0; th < 2; ++th) { for (int k = 0; k < 6; ++k) printf("%ld ", count(th, k)); }
          printf("%ld %ld\n", seen(0), seen(1)); return 0; }
    '''))
    exe = tmp_path / "probe"
    subprocess.run(["cc", "-O0", "-o", str(exe), str(src), "-ldl", "-lpthread"], check=True)
    out = subprocess.run([str(exe)], env={"LD_PRELOAD": str(shim)}, capture_output=True, text=True, check=True)
    values = list(map(int, out.stdout.split()))
    kinds = len(hotpath_shim.KINDS)
    counts = {th: dict(zip(hotpath_shim.KINDS, values[i * kinds:(i + 1) * kinds]))
              for i, th in enumerate(hotpath_shim.THREADS)}
    expected = {"malloc": 3, "free": 3, "mutex": 2, "cond": 1, "clock": 1, "sleep": 1}
    assert counts == {"service": expected, "copy": expected}, counts
    # Two ram-miss threads were recognized (the disarmed one too), so its zero is "not counted", not "not seen".
    assert values[2 * kinds:] == [2, 1], values


def test_characterize_the_hot_path(shim, tmp_path):
    """Prints the per-request counts of the build the service loads, so the baseline (master) and every later phase
    can be compared. It asserts that the measurement happened -- the requested window was measured, every kind was
    read for both threads, and the shim recognized exactly one service and one copy thread by name -- not what the
    counts are; the zero-count tests of later tasks do that."""
    counts = hotpath_shim.run_child(shim, requests=REQUESTS, tmp=tmp_path)
    assert counts["requests"] == REQUESTS
    for th in hotpath_shim.THREADS:
        assert set(counts[th]) == set(hotpath_shim.KINDS), (th, counts[th])
    # A shim that never recognized a thread reports all zeros; that must fail here, not pass as "zero overhead".
    assert counts["threads"] == {"service": 1, "copy": 1}, counts["threads"]
    per = {th: {k: round(v / counts["requests"], 2) for k, v in counts[th].items()} for th in hotpath_shim.THREADS}
    print("HOTPATH raw", counts)
    print("HOTPATH per request", per)
