// Completion-word probe (phase 2 Task P1 of plan 2026-09-29-hotpath-zero-overhead). Evidence for replacing the copy
// thread's cuEventQuery polling with a word the GPU writes into host-mapped pinned memory after a job's copies, which
// the copy thread reads with a plain acquire load. Not product code; built and run by run_probe.sh on divix01.
//
// What it reports:
//   1. Device attributes: every stream-mem-op attribute the installed cuda.h defines (_V1 and current), host mapping.
//   2. Per mechanism, over N jobs of E cuMemcpyAsync (H2D, registered host memory -> device) each, on a
//      CudaCopyBackend-like stream (non-blocking, greatest priority), run on a thread named "probe-copy-eng" so the
//      LD_PRELOAD counting shim (python/sglang/test/hotpath_shim.c) counts it as the copy thread:
//        ev1   cuEventRecord after the copies, cuEventQuery every poll turn
//        ev8   the same, one query every 8 turns (the branch's CopyEngine::run cadence, kQueryEvery)
//        wv32  (a) cuStreamWriteValue32(host-mapped word, seq) after the copies; poll = acquire load
//        wv64  (a) cuStreamWriteValue64
//        kern  (b) cuLaunchKernel of a 1-thread kernel doing a system-scope release store of seq to the word
//        ctl   negative control: cuStreamWriteValue32 on a SECOND stream (not ordered after the copies)
//      For each: per job driver calls, libcuda mutex/malloc on the submitting thread split into the copies and the
//      completion op, mutex/malloc while polling, and submit-end -> host-observes latency.
//      Stream order: after observing completion, the last 4 KiB of the job's last copy are read back (cuMemcpyDtoH,
//      legacy stream, which a non-blocking stream does not sync with) and compared with the pattern written into the
//      source for this job. `ctl` must show violations on large jobs, or the check has no power.
//   3. Dual observers (the direct "copy done -> host sees" comparison): the completion op and a cuEventRecord are
//      both enqueued after the copies, in either order; this thread polls the word, a second thread polls
//      cuEventQuery every turn; the difference of their observation times is reported.
//   4. Host read cost: ns per acquire load of the word, and the shim's counts over 10^7 loads (must be 0).
//   5. Lifecycle: CudaCopyBackend::init's and shutdown's calls on the tracked thread, counted (start-up/exit only),
//      plus the completion word's own setup (cuMemHostAlloc DEVICEMAP + cuMemHostGetDevicePointer).
//   6. --fault (a separate process): the word's write targets an unmapped device address. The word never arrives; the
//      poll loop's turn-budgeted liveness check (one cuStreamQuery per budget of turns, no clock on the poll path)
//      must see the stream's error and fail closed.
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                              \
  do {                                                                                                     \
    CUresult r_ = (x);                                                                                     \
    if (r_ != CUDA_SUCCESS) {                                                                              \
      const char* s_ = nullptr;                                                                            \
      cuGetErrorName(r_, &s_);                                                                             \
      fprintf(stderr, "%s:%d %s -> %d %s\n", __FILE__, __LINE__, #x, static_cast<int>(r_), s_ ? s_ : "?"); \
      exit(2);                                                                                             \
    }                                                                                                      \
  } while (0)

// rdtsc / pause as inline asm: nvcc's front end cannot parse <x86intrin.h> with this host compiler.
static inline uint64_t rdtsc() {
  uint32_t lo, hi;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return (static_cast<uint64_t>(hi) << 32) | lo;
}
static inline void cpu_pause() {
  asm volatile("pause" ::: "memory");
}

__global__ void release_store_kernel(unsigned long long* word, unsigned long long value) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> w(*word);
  w.store(value, cuda::memory_order_release);
}

// ---- the counting shim, when preloaded (role 1 = the copy thread: a pthread name ending "-copy-eng") ----
using CountFn = long (*)(int, int);
static CountFn shim_count = nullptr;
struct Counts {
  long malloc_ = 0, free_ = 0, mutex = 0, cond = 0, futex = 0, clock = 0;
  Counts operator-(const Counts& o) const {
    return {malloc_ - o.malloc_, free_ - o.free_, mutex - o.mutex, cond - o.cond, futex - o.futex, clock - o.clock};
  }
  Counts& operator+=(const Counts& o) {
    malloc_ += o.malloc_, free_ += o.free_, mutex += o.mutex, cond += o.cond, futex += o.futex, clock += o.clock;
    return *this;
  }
};
static Counts snap() {
  if (shim_count == nullptr) return {};
  return {shim_count(1, 0), shim_count(1, 1), shim_count(1, 2), shim_count(1, 3), shim_count(1, 6),
          shim_count(1, 4)};  // kinds: malloc, free, mutex, cond, futex, clock (clock_gettime)
}

static double ns_per_tick = 1.0;
static void calibrate_tsc() {
  timespec a{}, b{};
  clock_gettime(CLOCK_MONOTONIC, &a);
  const uint64_t t0 = rdtsc();
  timespec d{0, 200'000'000};
  nanosleep(&d, nullptr);
  clock_gettime(CLOCK_MONOTONIC, &b);
  const uint64_t t1 = rdtsc();
  ns_per_tick = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / static_cast<double>(t1 - t0);
}
static double us(uint64_t ticks) {
  return ticks * ns_per_tick / 1e3;
}

struct Pctl {
  double p50 = 0, p90 = 0, p99 = 0, max = 0, mean = 0;
};
static Pctl pctl(std::vector<double> v) {
  Pctl p;
  if (v.empty()) return p;
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))]; };
  p.p50 = at(0.5), p.p90 = at(0.9), p.p99 = at(0.99), p.max = v.back();
  double s = 0;
  for (double x : v) s += x;
  p.mean = s / v.size();
  return p;
}

// ---- shared state ----
static CUdevice dev;
static CUcontext ctx;
static CUstream st = nullptr, st2 = nullptr;
static CUevent evs[64];
static CUfunction kern_fn;
static uint8_t* src = nullptr;  // registered host memory (mmap + cuMemHostRegister), like production's slabs
static CUdeviceptr dst = 0;
static uint8_t* chk = nullptr;  // pinned readback buffer
static volatile uint32_t* w32 = nullptr;
static volatile uint64_t* w64 = nullptr;
static volatile uint64_t* wk = nullptr;
static CUdeviceptr dw32 = 0, dw64 = 0, dwk = 0;
static constexpr size_t kCheck = 4096;
static constexpr int kEntries = 12;  // ~2.1 lanes x at most 6 table entries per production job (results.md 8e)
static constexpr size_t kMaxEntryBytes = 1 << 20;

static FILE* json_out = nullptr;
static bool first_json = true;
static void jprint(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void jprint(const char* fmt, ...) {
  va_list a;
  va_start(a, fmt);
  vfprintf(json_out, fmt, a);
  va_end(a);
}
static void jsep() {
  if (!first_json) jprint(",\n");
  first_json = false;
}

enum Mech { EV1, EV8, WV32, WV64, KERN, CTL, kMechs };
static const char* kMechName[kMechs] = {"ev1", "ev8", "wv32", "wv64", "kern", "ctl"};

static void fill_pattern(size_t entry_bytes, uint32_t seq) {
  uint32_t* p = reinterpret_cast<uint32_t*>(src + (kEntries - 1) * entry_bytes + entry_bytes - kCheck);
  for (size_t i = 0; i < kCheck / 4; ++i) p[i] = seq * 2654435761u + static_cast<uint32_t>(i);
}
static bool check_pattern(size_t entry_bytes, uint32_t seq) {
  CK(cuMemcpyDtoH(chk, dst + (kEntries - 1) * entry_bytes + entry_bytes - kCheck, kCheck));
  const uint32_t* p = reinterpret_cast<const uint32_t*>(chk);
  for (size_t i = 0; i < kCheck / 4; ++i)
    if (p[i] != seq * 2654435761u + static_cast<uint32_t>(i)) return false;
  return true;
}

static void issue_copies(size_t entry_bytes) {
  for (int e = 0; e < kEntries; ++e) CK(cuMemcpyAsync(dst + e * entry_bytes, reinterpret_cast<CUdeviceptr>(src + e * entry_bytes), entry_bytes, st));
}

static void issue_op(Mech m, uint64_t seq) {
  switch (m) {
    case EV1:
    case EV8:
      CK(cuEventRecord(evs[seq % 64], st));
      break;
    case WV32:
      CK(cuStreamWriteValue32(st, dw32, static_cast<cuuint32_t>(seq), CU_STREAM_WRITE_VALUE_DEFAULT));
      break;
    case WV64:
      CK(cuStreamWriteValue64(st, dw64, seq, CU_STREAM_WRITE_VALUE_DEFAULT));
      break;
    case KERN: {
      void* args[] = {&dwk, &seq};
      CK(cuLaunchKernel(kern_fn, 1, 1, 1, 1, 1, 1, 0, st, args, nullptr));
      break;
    }
    case CTL:
      CK(cuStreamWriteValue32(st2, dw32, static_cast<cuuint32_t>(seq), CU_STREAM_WRITE_VALUE_DEFAULT));
      break;
    default:
      break;
  }
}

static bool word_done(Mech m, uint64_t seq) {
  switch (m) {
    case WV32:
    case CTL:
      return __atomic_load_n(w32, __ATOMIC_ACQUIRE) == static_cast<uint32_t>(seq);
    case WV64:
      return __atomic_load_n(w64, __ATOMIC_ACQUIRE) == seq;
    case KERN:
      return __atomic_load_n(wk, __ATOMIC_ACQUIRE) == seq;
    default:
      return false;
  }
}

static uint64_t g_seq = 0;  // job sequence, strictly increasing over the whole run (words never repeat a value)

// One mechanism, `jobs` measured jobs (after `warm` uncounted ones), on the calling (tracked) thread.
static void run_mech(Mech m, size_t entry_bytes, int jobs, int warm, const char* size_label) {
  std::vector<double> lat, sub_us, copies_us, op_us;
  Counts copies_c, op_c, poll_c;
  long queries = 0, turns = 0, violations = 0;
  for (int j = 0; j < warm + jobs; ++j) {
    const uint64_t seq = ++g_seq;
    fill_pattern(entry_bytes, static_cast<uint32_t>(seq));
    const Counts a = snap();
    const uint64_t t0 = rdtsc();
    issue_copies(entry_bytes);
    const uint64_t t1 = rdtsc();
    const Counts b = snap();
    issue_op(m, seq);
    const uint64_t t2 = rdtsc();
    const Counts c = snap();
    long q = 0, t = 0;
    if (m == EV1 || m == EV8) {
      const uint32_t every = m == EV1 ? 1 : 8;
      while (true) {
        ++t;
        if (t % every == 0) {
          ++q;
          const CUresult r = cuEventQuery(evs[seq % 64]);
          if (r == CUDA_SUCCESS) break;
          if (r != CUDA_ERROR_NOT_READY) CK(r);
        }
        cpu_pause();
      }
    } else {
      while (!word_done(m, seq)) {
        ++t;
        cpu_pause();
      }
    }
    const uint64_t t3 = rdtsc();
    const Counts d = snap();
    const bool ok = check_pattern(entry_bytes, static_cast<uint32_t>(seq));
    CK(cuStreamSynchronize(st));
    if (m == CTL) CK(cuStreamSynchronize(st2));
    if (j < warm) continue;
    copies_c += b - a;
    op_c += c - b;
    poll_c += d - c;
    queries += q;
    turns += t;
    violations += ok ? 0 : 1;
    lat.push_back(us(t3 - t2));
    sub_us.push_back(us(t2 - t0));
    copies_us.push_back(us(t1 - t0));
    op_us.push_back(us(t2 - t1));
  }
  const Pctl L = pctl(lat), S = pctl(sub_us), C = pctl(copies_us), O = pctl(op_us);
  const double n = jobs;
  printf("%-5s %-5s jobs %4d | per job: copies %d calls mutex %.2f malloc %.2f | op mutex %.2f malloc %.2f | "
         "poll: queries %.1f mutex %.1f malloc %.2f | submit->seen us p50 %.1f p90 %.1f p99 %.1f max %.1f | "
         "submit us p50 %.2f (copies %.2f, op %.2f) | order violations %ld\n",
         kMechName[m], size_label, jobs, kEntries, copies_c.mutex / n, copies_c.malloc_ / n, op_c.mutex / n,
         op_c.malloc_ / n, queries / n, poll_c.mutex / n, poll_c.malloc_ / n, L.p50, L.p90, L.p99, L.max, S.p50,
         C.p50, O.p50, violations);
  printf("      clock_gettime per job: copies %.2f, op %.2f, poll %.2f (the zero rule's clock kind)\n", copies_c.clock / n,
         op_c.clock / n, poll_c.clock / n);
  jsep();
  jprint("  {\"test\": \"mech\", \"mech\": \"%s\", \"size\": \"%s\", \"entry_bytes\": %zu, \"entries\": %d, "
         "\"jobs\": %d, \"copies_mutex_per_job\": %.4f, \"copies_malloc_per_job\": %.4f, \"copies_free_per_job\": %.4f, "
         "\"op_mutex_per_job\": %.4f, \"op_malloc_per_job\": %.4f, \"op_free_per_job\": %.4f, "
         "\"poll_queries_per_job\": %.2f, \"poll_turns_per_job\": %.1f, \"poll_mutex_per_job\": %.2f, "
         "\"poll_malloc_per_job\": %.4f, \"poll_futex_per_job\": %.4f, \"cond_per_job\": %.4f, "
         "\"latency_us\": {\"p50\": %.2f, \"p90\": %.2f, \"p99\": %.2f, \"max\": %.2f, \"mean\": %.2f}, "
         "\"submit_us_p50\": %.3f, \"copies_us_p50\": %.3f, \"op_us_p50\": %.3f, \"op_us_p99\": %.3f, "
         "\"copies_clock_per_job\": %.4f, \"op_clock_per_job\": %.4f, \"poll_clock_per_job\": %.4f, "
         "\"order_violations\": %ld}",
         kMechName[m], size_label, entry_bytes, kEntries, jobs, copies_c.mutex / n, copies_c.malloc_ / n,
         copies_c.free_ / n, op_c.mutex / n, op_c.malloc_ / n, op_c.free_ / n, queries / n, turns / n,
         poll_c.mutex / n, poll_c.malloc_ / n, poll_c.futex / n, (copies_c.cond + op_c.cond + poll_c.cond) / n, L.p50,
         L.p90, L.p99, L.max, L.mean, S.p50, C.p50, O.p50, O.p99, copies_c.clock / n, op_c.clock / n, poll_c.clock / n,
         violations);
}

// ---- dual observers: the word (this thread) against cuEventQuery every turn (a helper thread) ----
static std::atomic<uint64_t> dual_go{0};
static std::atomic<uint64_t> dual_seen_tick{0};
static std::atomic<bool> dual_stop{false};
static CUevent dual_ev = nullptr;

static void dual_helper() {
  pthread_setname_np(pthread_self(), "probe-evq");  // untracked by the shim
  CK(cuCtxSetCurrent(ctx));
  uint64_t served = 0;
  while (!dual_stop.load(std::memory_order_acquire)) {
    const uint64_t go = dual_go.load(std::memory_order_acquire);
    if (go == served) {
      cpu_pause();
      continue;
    }
    while (true) {
      const CUresult r = cuEventQuery(dual_ev);
      if (r == CUDA_SUCCESS) break;
      if (r != CUDA_ERROR_NOT_READY) CK(r);
      cpu_pause();
    }
    dual_seen_tick.store(rdtsc(), std::memory_order_release);
    served = go;
  }
}

// `event_first`: the event is recorded before the completion op (else after it).
static void run_dual(Mech m, bool event_first, size_t entry_bytes, int jobs, int warm, const char* size_label) {
  std::vector<double> diff;  // event seen - word seen, us
  for (int j = 0; j < warm + jobs; ++j) {
    const uint64_t seq = ++g_seq;
    dual_seen_tick.store(0, std::memory_order_relaxed);
    issue_copies(entry_bytes);
    if (event_first) CK(cuEventRecord(dual_ev, st));
    issue_op(m, seq);
    if (!event_first) CK(cuEventRecord(dual_ev, st));
    dual_go.store(seq, std::memory_order_release);
    while (!word_done(m, seq)) cpu_pause();
    const uint64_t tw = rdtsc();
    uint64_t te = 0;
    while ((te = dual_seen_tick.load(std::memory_order_acquire)) == 0) cpu_pause();
    CK(cuStreamSynchronize(st));
    if (j < warm) continue;
    diff.push_back((static_cast<double>(te) - static_cast<double>(tw)) * ns_per_tick / 1e3);
  }
  const Pctl D = pctl(diff);
  printf("dual  %-5s %-5s %s: event seen - word seen, us: p50 %.2f p90 %.2f p99 %.2f mean %.2f (n %d)\n",
         kMechName[m], size_label, event_first ? "event-then-op" : "op-then-event", D.p50, D.p90, D.p99, D.mean,
         jobs);
  jsep();
  jprint("  {\"test\": \"dual\", \"mech\": \"%s\", \"size\": \"%s\", \"order\": \"%s\", \"jobs\": %d, "
         "\"event_minus_word_us\": {\"p50\": %.3f, \"p90\": %.3f, \"p99\": %.3f, \"mean\": %.3f, \"min\": %.3f}}",
         kMechName[m], size_label, event_first ? "event_then_op" : "op_then_event", jobs, D.p50, D.p90, D.p99, D.mean,
         diff.empty() ? 0.0 : *std::min_element(diff.begin(), diff.end()));
}

static void host_read_cost() {
  const Counts a = snap();
  const uint64_t t0 = rdtsc();
  uint64_t acc = 0;
  constexpr long kLoads = 10'000'000;
  for (long i = 0; i < kLoads; ++i) acc += __atomic_load_n(w64, __ATOMIC_ACQUIRE);
  const uint64_t t1 = rdtsc();
  const Counts d = snap() - a;
  const double ns = (t1 - t0) * ns_per_tick / kLoads;
  printf("host read: %.2f ns per acquire load of the host-mapped word over %ld loads (acc %llu); shim over the loop: "
         "mutex %ld malloc %ld free %ld futex %ld\n",
         ns, kLoads, static_cast<unsigned long long>(acc & 1), d.mutex, d.malloc_, d.free_, d.futex);
  jsep();
  jprint("  {\"test\": \"host_read\", \"loads\": %ld, \"ns_per_load\": %.3f, \"mutex\": %ld, \"malloc\": %ld, "
         "\"free\": %ld, \"futex\": %ld, \"driver_calls\": 0}",
         kLoads, ns, d.mutex, d.malloc_, d.free_, d.futex);
}

// Per-call cost of cuEventQuery on a completed event, and of cuStreamQuery on an idle stream.
static void query_cost() {
  CK(cuEventRecord(evs[0], st));
  CK(cuStreamSynchronize(st));
  constexpr int kN = 100000;
  Counts a = snap();
  uint64_t t0 = rdtsc();
  for (int i = 0; i < kN; ++i) CK(cuEventQuery(evs[0]));
  uint64_t t1 = rdtsc();
  Counts d = snap() - a;
  printf("cuEventQuery (complete): %.1f ns/call, mutex %.3f malloc %.3f per call\n", (t1 - t0) * ns_per_tick / kN,
         d.mutex / double(kN), d.malloc_ / double(kN));
  jsep();
  jprint("  {\"test\": \"call_cost\", \"call\": \"cuEventQuery\", \"ns\": %.2f, \"mutex\": %.4f, \"malloc\": %.4f}",
         (t1 - t0) * ns_per_tick / kN, d.mutex / double(kN), d.malloc_ / double(kN));
  a = snap();
  t0 = rdtsc();
  for (int i = 0; i < kN; ++i) CK(cuStreamQuery(st));
  t1 = rdtsc();
  d = snap() - a;
  printf("cuStreamQuery (idle): %.1f ns/call, mutex %.3f malloc %.3f clock %.3f per call\n",
         (t1 - t0) * ns_per_tick / kN, d.mutex / double(kN), d.malloc_ / double(kN), d.clock / double(kN));
  jsep();
  jprint("  {\"test\": \"call_cost\", \"call\": \"cuStreamQuery\", \"ns\": %.2f, \"mutex\": %.4f, \"malloc\": %.4f}",
         (t1 - t0) * ns_per_tick / kN, d.mutex / double(kN), d.malloc_ / double(kN));
}

// CudaCopyBackend::init / shutdown's calls, on the tracked thread (copy_engine.h), and the word's own setup.
static Counts lifecycle_init() {
  const Counts a = snap();
  CK(cuInit(0));
  CUdevice d;
  CK(cuDeviceGet(&d, 0));
  CUcontext c;
  CK(cuDevicePrimaryCtxRetain(&c, d));
  CK(cuCtxSetCurrent(c));
  int least = 0, greatest = 0;
  CK(cuCtxGetStreamPriorityRange(&least, &greatest));
  CK(cuStreamCreateWithPriority(&st, CU_STREAM_NON_BLOCKING, greatest));
  for (auto& e : evs) CK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
  const Counts b = snap();
  // The completion word's setup: one pinned, device-mapped page (init only).
  void* page = nullptr;
  CK(cuMemHostAlloc(&page, 4096, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
  CUdeviceptr dpage = 0;
  CK(cuMemHostGetDevicePointer(&dpage, page, 0));
  const Counts e = snap();
  memset(page, 0, 4096);
  w32 = reinterpret_cast<volatile uint32_t*>(page);
  w64 = reinterpret_cast<volatile uint64_t*>(static_cast<uint8_t*>(page) + 64);
  wk = reinterpret_cast<volatile uint64_t*>(static_cast<uint8_t*>(page) + 128);
  dw32 = dpage, dw64 = dpage + 64, dwk = dpage + 128;
  const Counts bi = b - a, wi = e - b;
  printf("lifecycle init (CudaCopyBackend::init's calls): mutex %ld malloc %ld free %ld cond %ld | word setup "
         "(cuMemHostAlloc + cuMemHostGetDevicePointer): mutex %ld malloc %ld free %ld\n",
         bi.mutex, bi.malloc_, bi.free_, bi.cond, wi.mutex, wi.malloc_, wi.free_);
  jsep();
  jprint("  {\"test\": \"lifecycle_init\", \"backend\": {\"mutex\": %ld, \"malloc\": %ld, \"free\": %ld, \"cond\": %ld}, "
         "\"word_setup\": {\"mutex\": %ld, \"malloc\": %ld, \"free\": %ld}}",
         bi.mutex, bi.malloc_, bi.free_, bi.cond, wi.mutex, wi.malloc_, wi.free_);
  return bi;
}

static void lifecycle_shutdown() {
  const Counts a = snap();
  for (auto& e : evs) CK(cuEventDestroy(e));
  CK(cuStreamDestroy(st));
  CK(cuDevicePrimaryCtxRelease(dev));
  const Counts d = snap() - a;
  printf("lifecycle shutdown (CudaCopyBackend::shutdown's calls): mutex %ld malloc %ld free %ld\n", d.mutex, d.malloc_,
         d.free_);
  jsep();
  jprint("  {\"test\": \"lifecycle_shutdown\", \"mutex\": %ld, \"malloc\": %ld, \"free\": %ld}", d.mutex, d.malloc_,
         d.free_);
}

static void tracked_thread(int jobs_small, int jobs_large) {
  pthread_setname_np(pthread_self(), "probe-copy-eng");  // the shim's copy role
  lifecycle_init();
  CK(cuStreamCreateWithPriority(&st2, CU_STREAM_NON_BLOCKING, 0));
  CK(cuEventCreate(&dual_ev, CU_EVENT_DISABLE_TIMING));
  const struct {
    const char* label;
    size_t bytes;
    int jobs;
  } sizes[] = {{"tiny", 4096, jobs_small}, {"large", 512 * 1024, jobs_large}};
  for (const auto& s : sizes)
    for (int m = 0; m < kMechs; ++m) run_mech(static_cast<Mech>(m), s.bytes, s.jobs, 20, s.label);
  host_read_cost();
  query_cost();
  std::thread helper(dual_helper);
  for (const auto& s : sizes)
    for (Mech m : {WV32, WV64, KERN})
      for (bool ef : {false, true}) run_dual(m, ef, s.bytes, s.jobs, 20, s.label);
  dual_stop.store(true, std::memory_order_release);
  helper.join();
  CK(cuEventDestroy(dual_ev));
  CK(cuStreamDestroy(st2));
  lifecycle_shutdown();
}

// --fault: the completion op writes an unmapped device address; the host must fail closed without a clock.
static int run_fault(const char* how) {
  pthread_setname_np(pthread_self(), "probe-copy-eng");
  CK(cuCtxSetCurrent(ctx));
  int least = 0, greatest = 0;
  CK(cuCtxGetStreamPriorityRange(&least, &greatest));
  CK(cuStreamCreateWithPriority(&st, CU_STREAM_NON_BLOCKING, greatest));
  void* page = nullptr;
  CK(cuMemHostAlloc(&page, 4096, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
  memset(page, 0, 4096);
  w32 = reinterpret_cast<volatile uint32_t*>(page);
  const CUdeviceptr bad = 0x7f0000000000ull;  // no allocation there
  issue_copies(512 * 1024);
  CUresult enq = CUDA_SUCCESS;
  if (strcmp(how, "kern") == 0) {
    uint64_t seq = 1;
    CUdeviceptr p = bad;
    void* args[] = {&p, &seq};
    enq = cuLaunchKernel(kern_fn, 1, 1, 1, 1, 1, 1, 0, st, args, nullptr);
  } else {
    enq = cuStreamWriteValue32(st, bad, 1u, CU_STREAM_WRITE_VALUE_DEFAULT);
  }
  printf("fault %s: enqueue returned %d\n", how, static_cast<int>(enq));
  // The copy thread's poll with a turn budget, no clock: one cuStreamQuery per kBudget turns without the word.
  constexpr uint64_t kBudget = 1 << 16;  // ~2-3 ms of _mm_pause on this CPU
  uint64_t turns = 0, checks = 0;
  CUresult verdict = CUDA_SUCCESS;
  const uint64_t t0 = rdtsc();
  while (__atomic_load_n(w32, __ATOMIC_ACQUIRE) != 1u) {
    cpu_pause();
    if (++turns % kBudget != 0) continue;
    ++checks;
    const CUresult r = cuStreamQuery(st);
    if (r == CUDA_ERROR_NOT_READY) continue;
    if (r == CUDA_SUCCESS) {
      // The word was loaded before the query. A write that landed in between would make a healthy stream look like
      // a lost write, so re-load it (acquire) after the SUCCESS: the stream is idle, so its write, if made, is now
      // visible. Fail only if it is still short (P2 must test this interleaving).
      if (__atomic_load_n(w32, __ATOMIC_ACQUIRE) == 1u) break;
      verdict = CUDA_ERROR_UNKNOWN;  // idle stream, word still short after the re-load: a lost write -> fail stop
      break;
    }
    verdict = r;
    break;
  }
  const uint64_t t1 = rdtsc();
  const char* name = nullptr;
  cuGetErrorName(verdict, &name);
  printf("fault %s: word seen %s; liveness checks %llu, turns %llu, %.1f us to the verdict: %d %s -> %s\n", how,
         __atomic_load_n(w32, __ATOMIC_ACQUIRE) == 1u ? "yes" : "no", static_cast<unsigned long long>(checks),
         static_cast<unsigned long long>(turns), us(t1 - t0), static_cast<int>(verdict), name ? name : "?",
         verdict != CUDA_SUCCESS ? "FAIL CLOSED (leases kept, copy_failed)" : "no verdict");
  jsep();
  jprint("  {\"test\": \"fault\", \"how\": \"%s\", \"enqueue\": %d, \"word_seen\": %s, \"checks\": %llu, "
         "\"turns\": %llu, \"us_to_verdict\": %.1f, \"verdict\": %d, \"verdict_name\": \"%s\"}",
         how, static_cast<int>(enq), __atomic_load_n(w32, __ATOMIC_ACQUIRE) == 1u ? "true" : "false",
         static_cast<unsigned long long>(checks), static_cast<unsigned long long>(turns), us(t1 - t0),
         static_cast<int>(verdict), name ? name : "?");
  return verdict != CUDA_SUCCESS ? 0 : 1;
}

int main(int argc, char** argv) {
  const char* out = argc > 1 ? argv[1] : "probe.json";
  const char* fault = argc > 2 && strncmp(argv[2], "--fault=", 8) == 0 ? argv[2] + 8 : nullptr;
  const int jobs_small = argc > 2 && !fault ? atoi(argv[2]) : 2000;
  const int jobs_large = argc > 3 && !fault ? atoi(argv[3]) : 500;
  shim_count = reinterpret_cast<CountFn>(dlsym(RTLD_DEFAULT, "hotpath_shim_count"));
  json_out = fopen(out, "w");
  if (json_out == nullptr) return 3;
  jprint("{\"results\": [\n");
  calibrate_tsc();
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  int drv = 0;
  CK(cuDriverGetVersion(&drv));
  char name[256] = {};
  CK(cuDeviceGetName(name, sizeof(name), dev));
  // The runtime shares the primary context; it only resolves the kernel's CUfunction here.
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) return 4;
  cudaFunction_t fn = nullptr;
  if (cudaGetFuncBySymbol(&fn, reinterpret_cast<const void*>(release_store_kernel)) != cudaSuccess) return 5;
  kern_fn = reinterpret_cast<CUfunction>(fn);
  CK(cuCtxSetCurrent(ctx));
  void* libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  printf("device %s, driver API %d, shim %s, tsc %.4f ns/tick\n", name, drv, shim_count ? "preloaded" : "ABSENT",
         ns_per_tick);
  printf("libcuda exports: cuStreamWriteValue32 %s, cuStreamWriteValue32_v2 %s, cuStreamWriteValue64_v2 %s\n",
         libcuda && dlsym(libcuda, "cuStreamWriteValue32") ? "yes" : "no",
         libcuda && dlsym(libcuda, "cuStreamWriteValue32_v2") ? "yes" : "no",
         libcuda && dlsym(libcuda, "cuStreamWriteValue64_v2") ? "yes" : "no");
  jsep();
  jprint("  {\"test\": \"device\", \"name\": \"%s\", \"driver_api\": %d, \"shim\": %s, \"ns_per_tick\": %.5f", name, drv,
         shim_count ? "true" : "false", ns_per_tick);
  const struct {
    const char* n;
    CUdevice_attribute a;
  } attrs[] = {
      {"CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1", CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1},
      {"CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1", CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1},
      {"CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1", CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1},
      {"CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS", CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS},
      {"CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR", CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR},
      {"CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES", CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES},
      {"CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY", CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY},
      {"CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING", CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING},
      {"CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
       CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM},
      {"CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED", CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED},
      {"CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS", CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS},
  };
  jprint(", \"attributes\": {");
  for (size_t i = 0; i < sizeof(attrs) / sizeof(attrs[0]); ++i) {
    int v = -1;
    const CUresult r = cuDeviceGetAttribute(&v, attrs[i].a, dev);
    printf("  %-62s (%3d) = %d%s\n", attrs[i].n, static_cast<int>(attrs[i].a), v,
           r == CUDA_SUCCESS ? "" : " (query failed)");
    jprint("%s\"%s\": {\"id\": %d, \"value\": %d, \"rc\": %d}", i ? ", " : "", attrs[i].n, static_cast<int>(attrs[i].a),
           v, static_cast<int>(r));
  }
  jprint("}}");

  // Source: registered anonymous memory, like the tier's slabs; destination: device memory; readback: pinned.
  const size_t src_bytes = kEntries * kMaxEntryBytes;
  src = static_cast<uint8_t*>(mmap(nullptr, src_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (src == MAP_FAILED) return 6;
  memset(src, 0x5a, src_bytes);
  CK(cuMemHostRegister(src, src_bytes, CU_MEMHOSTREGISTER_PORTABLE));
  CK(cuMemAlloc(&dst, src_bytes));
  CK(cuMemHostAlloc(reinterpret_cast<void**>(&chk), kCheck, CU_MEMHOSTALLOC_PORTABLE));

  int rc = 0;
  if (fault != nullptr) {
    std::thread t([&] { rc = run_fault(fault); });
    t.join();
  } else {
    std::thread t([&] {
      CK(cuCtxSetCurrent(ctx));
      tracked_thread(jobs_small, jobs_large);
    });
    t.join();
  }
  jprint("\n]}\n");
  fclose(json_out);
  return rc;
}
