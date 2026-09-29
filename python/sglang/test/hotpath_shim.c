// LD_PRELOAD counting shim for plan 2026-09-29-hotpath-zero-overhead: counts allocator, mutex, condvar, clock,
// sleep and futex calls made by the RAM-miss service thread (a name ending "-ram-miss") and the copy-engine thread
// ("-copy-eng") while armed. Test only; never loaded in production.
//
// At ba01695c35 the two threads name themselves Layout::kName + "-ram-miss" (ram_thread.h, RamThread::run) and
// Layout::kName + "-copy-eng" (ram_tier.h -> copy_engine.h, CopyEngine::run), truncated to 15 bytes; for the exl3
// layout (kName "exl3") that is "exl3-ram-miss" and "exl3-copy-eng", 13 bytes each, so the suffix survives.
// `threads_seen` counts the matches whether or not the shim is armed, so a caller can tell "no calls" from "no
// thread was ever recognized".
//
// Whole-process mode (Task 18's arm C, a production server under LD_PRELOAD): when HOTPATH_SHIM_OUT is set, the
// constructor arms the shim at load, so every tracked thread is counted from the moment it names itself, and a
// destructor writes {"pid", "threads", "service", "copy"} as JSON to that path at exit -- only from a process that
// recognized at least one tracked thread (a server forks several processes, and all of them load the shim), and to
// "<path>.<pid>" when the path already exists, so no process overwrites another's counts. The destructor runs at a
// normal exit (exit(), which Python's own shutdown calls); a process killed by a signal writes nothing.
//
// What a zero from this shim does NOT rule out (the calls it cannot see):
//   - allocations libc makes internally (e.g. inside fopen, qsort, getaddrinfo or the dynamic loader), which call
//     the allocator without going through the interposed PLT symbols;
//   - pthread_rwlock_* and std::shared_mutex (not interposed), and any lock that is not a pthread mutex;
//   - futex calls that do not go through libc's syscall() (glibc's own lock internals, inline-asm syscalls,
//     std::atomic::wait/notify and std::counting_semaphore); only syscall(SYS_futex, ...) is counted, which is how
//     spsc_ring.h's futex_wait/futex_wake enter the kernel;
//   - clock reads that skip clock_gettime: rdtsc, the vDSO called directly, gettimeofday/time(), and clock_gettime
//     calls made from inside libc itself;
//   - sleeps other than nanosleep/clock_nanosleep (usleep, sched_yield, poll/epoll/select timeouts).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

enum { kMalloc, kFree, kMutex, kCond, kClock, kSleep, kFutex, kKinds };
static _Atomic long counts[2][kKinds];
static _Atomic long threads_seen[2];
static _Atomic int armed;
// initial-exec: the preloaded object's TLS lives in the static block, so reading `who` inside malloc never goes
// through __tls_get_addr (which may itself allocate for dynamically loaded modules).
static __thread int who __attribute__((tls_model("initial-exec"))) = -1;  // 0 service, 1 copy engine, -1 untracked

#define COUNT(kind) do { if (who >= 0 && atomic_load_explicit(&armed, memory_order_relaxed)) \
    atomic_fetch_add_explicit(&counts[who][kind], 1, memory_order_relaxed); } while (0)

extern void* __libc_malloc(size_t);
extern void* __libc_calloc(size_t, size_t);
extern void* __libc_realloc(void*, size_t);
extern void* __libc_memalign(size_t, size_t);
extern void __libc_free(void*);

void* malloc(size_t n) { COUNT(kMalloc); return __libc_malloc(n); }
void* calloc(size_t a, size_t b) { COUNT(kMalloc); return __libc_calloc(a, b); }
void* realloc(void* p, size_t n) { COUNT(kMalloc); return __libc_realloc(p, n); }
void* memalign(size_t a, size_t n) { COUNT(kMalloc); return __libc_memalign(a, n); }
void* aligned_alloc(size_t a, size_t n) { COUNT(kMalloc); return __libc_memalign(a, n); }
int posix_memalign(void** out, size_t a, size_t n) {
  COUNT(kMalloc);
  void* p = __libc_memalign(a, n);
  if (p == NULL) return 12;  // ENOMEM
  *out = p;
  return 0;
}
void free(void* p) { if (p) COUNT(kFree); __libc_free(p); }

// Resolved once, in a constructor, before any tracked thread exists: a lazy dlsym on a tracked thread could allocate
// (dlerror's buffer) and be counted as the hot path's.
static int (*real_mutex_lock)(pthread_mutex_t*);
static int (*real_mutex_trylock)(pthread_mutex_t*);
static int (*real_cond_wait)(pthread_cond_t*, pthread_mutex_t*);
static int (*real_cond_timedwait)(pthread_cond_t*, pthread_mutex_t*, const struct timespec*);
static int (*real_cond_clockwait)(pthread_cond_t*, pthread_mutex_t*, clockid_t, const struct timespec*);
static int (*real_cond_signal)(pthread_cond_t*);
static int (*real_cond_broadcast)(pthread_cond_t*);
static int (*real_clock_gettime)(clockid_t, struct timespec*);
static int (*real_nanosleep)(const struct timespec*, struct timespec*);
static int (*real_clock_nanosleep)(clockid_t, int, const struct timespec*, struct timespec*);
static int (*real_setname)(pthread_t, const char*);
static long (*real_syscall)(long, ...);
static char out_path[4096];  // HOTPATH_SHIM_OUT, copied at load; empty: no exit dump

__attribute__((constructor)) static void resolve(void) {
  real_mutex_lock = dlsym(RTLD_NEXT, "pthread_mutex_lock");
  real_mutex_trylock = dlsym(RTLD_NEXT, "pthread_mutex_trylock");
  real_cond_wait = dlsym(RTLD_NEXT, "pthread_cond_wait");
  real_cond_timedwait = dlsym(RTLD_NEXT, "pthread_cond_timedwait");
  real_cond_clockwait = dlsym(RTLD_NEXT, "pthread_cond_clockwait");
  real_cond_signal = dlsym(RTLD_NEXT, "pthread_cond_signal");
  real_cond_broadcast = dlsym(RTLD_NEXT, "pthread_cond_broadcast");
  real_clock_gettime = dlsym(RTLD_NEXT, "clock_gettime");
  real_nanosleep = dlsym(RTLD_NEXT, "nanosleep");
  real_clock_nanosleep = dlsym(RTLD_NEXT, "clock_nanosleep");
  real_setname = dlsym(RTLD_NEXT, "pthread_setname_np");
  real_syscall = dlsym(RTLD_NEXT, "syscall");
  // Test code: a raw getenv, read once at load, before any tracked thread exists.
  const char* out = getenv("HOTPATH_SHIM_OUT");
  if (out != NULL && out[0] != '\0' && strlen(out) < sizeof(out_path) - 32) {
    strcpy(out_path, out);
    atomic_store(&armed, 1);
  }
}

// A hook still NULL when it is first called (a call made before the constructor ran, e.g. from another preloaded
// object's constructor) resolves itself then instead of calling through NULL. Such an early call is never on a
// tracked thread, so the dlsym it makes is not counted.
#define RESOLVE(fn, sym) do { if (fn == NULL) fn = dlsym(RTLD_NEXT, sym); } while (0)

int pthread_mutex_lock(pthread_mutex_t* m) { COUNT(kMutex); RESOLVE(real_mutex_lock, "pthread_mutex_lock"); return real_mutex_lock(m); }
int pthread_mutex_trylock(pthread_mutex_t* m) { COUNT(kMutex); RESOLVE(real_mutex_trylock, "pthread_mutex_trylock"); return real_mutex_trylock(m); }
int pthread_cond_wait(pthread_cond_t* c, pthread_mutex_t* m) { COUNT(kCond); RESOLVE(real_cond_wait, "pthread_cond_wait"); return real_cond_wait(c, m); }
int pthread_cond_timedwait(pthread_cond_t* c, pthread_mutex_t* m, const struct timespec* t) { COUNT(kCond); RESOLVE(real_cond_timedwait, "pthread_cond_timedwait"); return real_cond_timedwait(c, m, t); }
int pthread_cond_clockwait(pthread_cond_t* c, pthread_mutex_t* m, clockid_t k, const struct timespec* t) { COUNT(kCond); RESOLVE(real_cond_clockwait, "pthread_cond_clockwait"); return real_cond_clockwait(c, m, k, t); }
int pthread_cond_signal(pthread_cond_t* c) { COUNT(kCond); RESOLVE(real_cond_signal, "pthread_cond_signal"); return real_cond_signal(c); }
int pthread_cond_broadcast(pthread_cond_t* c) { COUNT(kCond); RESOLVE(real_cond_broadcast, "pthread_cond_broadcast"); return real_cond_broadcast(c); }
int clock_gettime(clockid_t k, struct timespec* t) { COUNT(kClock); RESOLVE(real_clock_gettime, "clock_gettime"); return real_clock_gettime(k, t); }
int nanosleep(const struct timespec* a, struct timespec* b) { COUNT(kSleep); RESOLVE(real_nanosleep, "nanosleep"); return real_nanosleep(a, b); }
int clock_nanosleep(clockid_t k, int f, const struct timespec* a, struct timespec* b) { COUNT(kSleep); RESOLVE(real_clock_nanosleep, "clock_nanosleep"); return real_clock_nanosleep(k, f, a, b); }

// syscall(2) takes up to six word-sized arguments after the number; forwarding all six is what glibc's own wrapper
// reads, whatever the call. Only SYS_futex is counted (the copy engine's futex_wait on the copy thread, futex_wake on
// the service thread).
long syscall(long number, ...) {
  va_list args;
  va_start(args, number);
  long a[6];
  for (int i = 0; i < 6; ++i) a[i] = va_arg(args, long);
  va_end(args);
  if (number == SYS_futex) COUNT(kFutex);
  RESOLVE(real_syscall, "syscall");
  return real_syscall(number, a[0], a[1], a[2], a[3], a[4], a[5]);
}

int pthread_setname_np(pthread_t thread, const char* name) {
  if (pthread_equal(thread, pthread_self())) {
    size_t n = strlen(name);
    if (n >= 9 && strcmp(name + n - 9, "-ram-miss") == 0) {
      who = 0;
      atomic_fetch_add(&threads_seen[0], 1);
    } else if (n >= 9 && strcmp(name + n - 9, "-copy-eng") == 0) {
      who = 1;
      atomic_fetch_add(&threads_seen[1], 1);
    }
  }
  RESOLVE(real_setname, "pthread_setname_np");
  return real_setname(thread, name);
}

void hotpath_shim_arm(int on) { atomic_store(&armed, on); }
void hotpath_shim_reset(void) { for (int t = 0; t < 2; ++t) for (int k = 0; k < kKinds; ++k) atomic_store(&counts[t][k], 0); }
long hotpath_shim_count(int thread, int kind) { return atomic_load(&counts[thread][kind]); }
long hotpath_shim_threads(int thread) { return atomic_load(&threads_seen[thread]); }

static const char* const kKindNames[kKinds] = {"malloc", "free", "mutex", "cond", "clock", "sleep", "futex"};

// The exit dump (HOTPATH_SHIM_OUT). Formats into a stack buffer and writes with open/write: no stdio, no allocation.
__attribute__((destructor)) static void dump(void) {
  if (out_path[0] == '\0') return;
  const long seen0 = atomic_load(&threads_seen[0]), seen1 = atomic_load(&threads_seen[1]);
  if (seen0 + seen1 == 0) return;  // not the process that ran the service: leave the file to the one that did
  char buf[1024];
  int n = snprintf(buf, sizeof(buf), "{\"pid\": %d, \"threads\": {\"service\": %ld, \"copy\": %ld}", (int)getpid(),
                   seen0, seen1);
  for (int t = 0; t < 2; ++t) {
    n += snprintf(buf + n, sizeof(buf) - n, ", \"%s\": {", t == 0 ? "service" : "copy");
    for (int k = 0; k < kKinds; ++k)
      n += snprintf(buf + n, sizeof(buf) - n, "%s\"%s\": %ld", k ? ", " : "", kKindNames[k],
                    atomic_load(&counts[t][k]));
    n += snprintf(buf + n, sizeof(buf) - n, "}");
  }
  n += snprintf(buf + n, sizeof(buf) - n, "}\n");
  int fd = open(out_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0 && errno == EEXIST) {
    char alt[sizeof(out_path) + 16];
    snprintf(alt, sizeof(alt), "%s.%d", out_path, (int)getpid());
    fd = open(alt, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  }
  if (fd < 0) return;
  for (int off = 0; off < n;) {
    ssize_t w = write(fd, buf + off, n - off);
    if (w <= 0) break;
    off += (int)w;
  }
  close(fd);
}
