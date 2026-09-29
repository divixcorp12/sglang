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
    """Self-test: a thread named x-ram-miss makes exactly 3 mallocs, 3 frees, 2 mutex locks and 1 clock read, and an
    unnamed thread's identical calls are not counted."""
    src = tmp_path / "probe.c"
    src.write_text(textwrap.dedent(r'''
        #define _GNU_SOURCE
        #include <pthread.h>
        #include <stdlib.h>
        #include <stdio.h>
        #include <time.h>
        #include <dlfcn.h>
        static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
        static void work(void) { struct timespec t; for (int i = 0; i < 3; ++i) free(malloc(64));
          pthread_mutex_lock(&m); pthread_mutex_unlock(&m); pthread_mutex_lock(&m); pthread_mutex_unlock(&m);
          clock_gettime(CLOCK_MONOTONIC, &t); }
        static void* named(void* a) { pthread_setname_np(pthread_self(), "x-ram-miss"); work(); return 0; }
        static void* plain(void* a) { work(); return 0; }
        int main(void) { void (*arm)(int) = dlsym(RTLD_DEFAULT, "hotpath_shim_arm");
          long (*count)(int, int) = dlsym(RTLD_DEFAULT, "hotpath_shim_count"); arm(1);
          pthread_t a, b; pthread_create(&a, 0, named, 0); pthread_join(a, 0);
          pthread_create(&b, 0, plain, 0); pthread_join(b, 0);
          printf("%ld %ld %ld %ld\n", count(0, 0), count(0, 1), count(0, 2), count(0, 4)); return 0; }
    '''))
    exe = tmp_path / "probe"
    subprocess.run(["cc", "-O0", "-o", str(exe), str(src), "-ldl", "-lpthread"], check=True)
    out = subprocess.run([str(exe)], env={"LD_PRELOAD": str(shim)}, capture_output=True, text=True, check=True)
    malloc, free, mutex, clock = map(int, out.stdout.split())
    assert (malloc, free, mutex, clock) == (3, 3, 2, 1)


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
