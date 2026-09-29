// AsyncFileReader over one io_uring ring shared by every model file.
#pragma once

#include "../../../io/registered_buffers.h"  // relative: the JIT build does not put jit/csrc on the include path
#include "build_policy.h"
#include "file_reader.h"
#include "uring_options.h"
#include <sys/resource.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <liburing.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(IO_URING_VERSION_MAJOR) && \
    (IO_URING_VERSION_MAJOR > 2 || (IO_URING_VERSION_MAJOR == 2 && IO_URING_VERSION_MINOR >= 10))
#define SGLANG_URING_HAS_READV_FIXED 1
#else
#define SGLANG_URING_HAS_READV_FIXED 0
#endif

namespace sglang::expert_stream {

// `Build` (build_policy.h) gates the reader's own test hooks (the ring-reset faults FaultyReader forwards, and
// publish_sq_without_enter): ProdBuild has neither the hooks nor their state (plan 2026-09-29-hotpath-zero-overhead
// Task 10). UringReader is the production reader; InstrUringReader is the one the instrumented build wraps in
// FaultyReader and the ring-reset harness drives.
template <class Build>
class BasicUringReader {
  static_assert(BuildPolicy<Build>);

 public:
  BasicUringReader() = default;
  BasicUringReader(const BasicUringReader&) = delete;
  BasicUringReader& operator=(const BasicUringReader&) = delete;
  ~BasicUringReader() {
    close();
  }

  bool init(unsigned depth) {
    close();
    options_ = UringOptions::from_env();
    depth_ = depth;
    if (depth == 0 || depth > 32768) throw std::invalid_argument("io_uring depth must be in [1, 32768]");
#if !SGLANG_URING_HAS_READV_FIXED
    if (options_.read_mode == UringReadMode::ReadvFixed)
      throw std::runtime_error("SGLANG_EXPERT_STREAM_URING_READ_MODE=readv_fixed requires liburing 2.10+ headers");
#endif
    return create_ring();
  }
  bool ready() const {
    return ready_;
  }

  // Buffers describe their complete live allocations (one per named slab, or the bounce), never a guessed interval
  // between unrelated slabs. Each is registered as row-aligned chunks of at most 1 GiB (io::RegisteredBufferTable).
  // Registration is independent of CUDA pinning. The owner must retain these allocations through close().
  void configure_resources(const std::vector<int>& fds, const std::vector<RegisteredRegion>& buffers, bool direct) {
    if (!ready_ || configured_ || outstanding_ != 0)
      throw std::logic_error("io_uring resources must be configured once on a ready, idle reader");
    if (options_.iopoll() && !direct)
      throw std::invalid_argument("SGLANG_EXPERT_STREAM_URING_MODE with iopoll requires O_DIRECT files");
    if (options_.fixed_files) {
      if (fds.empty()) throw std::invalid_argument("fixed files requested with an empty file table");
      for (int fd : fds)
        if (fd < 0) throw std::invalid_argument("fixed file table contains an invalid fd");
      files_ = fds;
      // O(1) fd -> registered index (mapped_fd), -1 for an fd not in the table.
      fd_index_.assign(static_cast<size_t>(*std::max_element(fds.begin(), fds.end())) + 1, -1);
      for (size_t i = 0; i < fds.size(); ++i)
        fd_index_[static_cast<size_t>(fds[i])] = static_cast<int>(i);
    }
    if (options_.read_mode != UringReadMode::Normal) {
      if (buffers.empty() || buffers.size() > 65536)
        throw std::invalid_argument("fixed reads require between 1 and 65536 registered buffer regions");
      for (const RegisteredRegion& b : buffers) {
        const auto start = reinterpret_cast<uintptr_t>(b.base);
        if (start == 0 || b.bytes == 0 || b.row_bytes == 0 || b.bytes % b.row_bytes != 0 ||
            b.bytes > std::numeric_limits<uintptr_t>::max() - start)
          throw std::invalid_argument("invalid registered buffer region");
      }
      regions_ = buffers;
    }
    register_resources();
    configured_ = true;
    diagnostics();
  }

  // Settles writes before unregistering resources. Closing an fd/ring alone does not prove DMA has stopped.
  void close() noexcept {
    if (ready_) {
      try {
        if (outstanding_ != 0) drain(outstanding_);
      } catch (...) {
        // A refused NOP drain throws after closing the ring with nothing left in the kernel: safe to go on. Anything
        // else might leave memory the kernel still owns, so never let the caller free it.
        if (ready_ || outstanding_ != 0) std::terminate();
      }
      close_ring();
      if constexpr (Build::kMetrics) {
        if (options_.diagnostics && fanout_.reads != 0) report_fixed();
      }
    }
    files_.clear();
    fd_index_.clear();
    regions_.clear();
    configured_ = false;
  }

  bool prep_read(int fd, void* buf, unsigned len, uint64_t off, uint64_t tag) {
    require_ready();
    if (fixed_reads()) throw std::logic_error("fixed read modes prepare through prep_readv_fixed");
    const int file = mapped_fd(fd);
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    io_uring_prep_read(sqe, file, buf, len, off);
    finish_prep(sqe, tag);
    return true;
  }

  bool prep_readv(int fd, const iovec* iov, unsigned count, uint64_t off, uint64_t tag) {
    require_ready();
    if (fixed_reads()) throw std::logic_error("fixed read modes prepare through prep_readv_fixed");
    if (iov == nullptr || count == 0) throw std::invalid_argument("readv requires a nonempty iovec array");
    const int file = mapped_fd(fd);
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    io_uring_prep_readv(sqe, file, iov, count, off);
    finish_prep(sqe, tag);
    return true;
  }

  // Fixed reads (READ_MODE fixed or readv_fixed). A read's iovecs are grouped into legs (fixed_legs) and each leg is
  // prepared by prep_readv_fixed with its registered buffer; the caller keeps `iov` alive until the leg is reaped.
  bool fixed_reads() const {
    return options_.read_mode != UringReadMode::Normal;
  }
  // Cut reads into device-sized legs (READ_CUTS; ReaderCore plans them): the resolved option, for diagnostics.
  bool read_cuts() const {
    return options_.read_cuts_on();
  }
  unsigned sq_space() const {
    return io_uring_sq_space_left(&ring_);
  }
  // Test only (fault word fixed_chunk_cap): a registration chunk cap below 1 GiB, so small slabs register as many
  // chunks. 0 restores the default.
  void set_fixed_chunk_cap(size_t cap) {
    if (configured_) throw std::logic_error("the fixed-buffer chunk cap is set before resources are configured");
    chunk_cap_ = cap == 0 ? sglang::io::kMaxRegisteredBufferBytes : cap;
  }
  // Consecutive iovecs sharing one registered buffer form a leg (READ_FIXED: one iovec per leg). Rows never straddle
  // buffers (plan_chunks), so every iovec lies in exactly one buffer and nothing is clipped.
  unsigned fixed_legs(const iovec* iov, unsigned count, FixedLeg* out) const {
    unsigned legs = 0;
    for (unsigned i = 0; i < count; ++i) {
      const int b = table_.find(reinterpret_cast<uint64_t>(iov[i].iov_base), iov[i].iov_len);
      if (b < 0) throw std::invalid_argument("a fixed read's destination lies in no registered buffer");
      if (legs > 0 && out[legs - 1].buffer == b && options_.read_mode == UringReadMode::ReadvFixed) {
        ++out[legs - 1].count;
        out[legs - 1].bytes += iov[i].iov_len;
      } else {
        out[legs++] = FixedLeg{i, 1, b, iov[i].iov_len};
      }
    }
    return legs;
  }
  // Counts a fixed read's legs (diagnostics only, spec M10): reads prepared, those cut into more than one leg, and
  // their SQEs. InstrBuild only; on ProdBuild the call is empty and the counters and their accessors do not exist.
  void note_fanout([[maybe_unused]] unsigned legs) {
    if constexpr (Build::kMetrics) {
      ++fanout_.reads;
      if (legs > 1) {
        ++fanout_.cuts;
        fanout_.sqes += legs;
      }
      if (options_.diagnostics && fanout_.reads >= fanout_.next_report) {
        fanout_.next_report <<= 1;
        report_fixed();
      }
    }
  }
  uint64_t fixed_cuts() const
    requires Build::kMetrics
  {
    return fanout_.cuts;
  }
  uint64_t fanout_sqes() const
    requires Build::kMetrics
  {
    return fanout_.sqes;
  }
  // How many times the ring's resources were registered: 1 per configure_resources, +1 per ring reset. Cold path
  // only (tests read it to prove a drain kept the registered tier).
  uint64_t registrations() const {
    return registrations_;
  }
  bool prep_readv_fixed(int fd, const iovec* iov, unsigned count, uint64_t off, int buffer, uint64_t tag) {
    require_ready();
    if (!fixed_reads()) throw std::logic_error("prep_readv_fixed needs a fixed read mode");
    if (iov == nullptr || count == 0) throw std::invalid_argument("a fixed read requires a nonempty iovec array");
    const bool scalar = options_.read_mode == UringReadMode::Fixed;
    if (scalar && count != 1)
      throw std::invalid_argument("READ_MODE=fixed supports one iovec per leg; use readv_fixed for scattered reads");
    if (scalar && iov[0].iov_len > std::numeric_limits<unsigned>::max())
      throw std::invalid_argument("fixed read length exceeds unsigned range");
    const int file = mapped_fd(fd);  // before taking an SQE: a refused fd must leave the SQ untouched
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    if (scalar) {
      io_uring_prep_read_fixed(sqe, file, iov[0].iov_base, static_cast<unsigned>(iov[0].iov_len), off, buffer);
    } else {
#if SGLANG_URING_HAS_READV_FIXED
      io_uring_prep_readv_fixed(sqe, file, iov, count, off, 0, buffer);
#else
      // init() refuses READ_MODE=readv_fixed without liburing 2.10 headers, so no ring reaches here.
      throw std::logic_error("READ_MODE=readv_fixed without IORING_OP_READV_FIXED support");
#endif
    }
    finish_prep(sqe, tag);
    return true;
  }

  int submit(unsigned wait_nr) {
    if (options_.blocking_wait())
      return wait_nr != 0 ? io_uring_submit_and_wait(&ring_, wait_nr) : io_uring_submit(&ring_);
    const int submitted = io_uring_submit(&ring_);
    if (submitted < 0 || wait_nr == 0) return submitted;
    while (io_uring_cq_ready(&ring_) < wait_nr) {
      // Short submissions must be retried: waiting on an SQE still in userspace cannot make progress.
      if (io_uring_sq_ready(&ring_) != 0) {
        const int rc = io_uring_submit(&ring_);
        if (rc < 0) return rc;
      }
      // Without SQPOLL, IOPOLL needs GETEVENTS to drive storage polling. With both flags, the kernel
      // SQ thread drives IOPOLL too; the application only observes its CQ and need not enter the kernel.
      if (options_.iopoll() && !options_.sqpoll()) {
        const int rc = io_uring_get_events(&ring_);
        if (rc < 0) return rc;
      }
      spin_hint();
    }
    return submitted;
  }

  unsigned reap(std::vector<ReadCompletion>& out) {
    io_uring_cqe* cqe;
    unsigned head;
    unsigned seen = 0;
    io_uring_for_each_cqe(&ring_, head, cqe) {
      out.push_back(ReadCompletion{io_uring_cqe_get_data64(cqe), cqe->res});
      ++seen;
    }
    io_uring_cq_advance(&ring_, seen);
    if (seen > outstanding_) std::terminate();
    outstanding_ -= seen;
    return seen;
  }

  void drain(unsigned pending) {
    if (pending != outstanding_) std::terminate();
    // A failed ring reset closed the ring (outstanding_ 0): nothing it held can still be in flight.
    if (!ready_) {
      if (outstanding_ != 0) std::terminate();
      return;
    }
    if (options_.sqpoll()) {
      // SQPOLL consumes concurrently: no snapshot may classify SQEs as safe to abandon.
      // Publish/retry EVERY prepared read, then retire EVERY completion before releasing buffers.
      while (outstanding_ > 0) {
        const int submitted = io_uring_submit(&ring_);
        if (submitted < 0 && !soft_error(submitted)) std::terminate();
        io_uring_cqe* cqe = nullptr;
        const int rc = io_uring_peek_cqe(&ring_, &cqe);
        if (rc == 0)
          retire(cqe);
        else if (rc != -EAGAIN && rc != -EINTR)
          std::terminate();
        else if (io_uring_sq_ready(&ring_) == 0)
          wait_one();
        else
          spin_hint();
      }
      return;
    }
    // No asynchronous SQ consumer: remaining SQEs can be discarded only AFTER all consumed reads retire.
    const unsigned unsubmitted = std::min(pending, io_uring_sq_ready(&ring_));
    while (outstanding_ > unsubmitted)
      wait_one();
    if (unsubmitted == 0) return;
    // Discard them on the same ring, as NOPs, so the registered buffer table and file table survive. Re-creating the
    // ring re-registered the whole tier: 59 s for 107 GB on divix01 (analysis/dsv41-drive/uring-reg/results.md),
    // past RamThread's 30 s fatal_wait, which aborts a request held in service that long.
    const int refused = flush_as_nops(unsubmitted);
    if (refused == 0) return;
    // The kernel refused the NOP submission. What is left never entered the kernel, so the ring can close now.
    close_ring();
    outstanding_ = 0;
    // A fixed read mode never resets: re-registering its buffers would outlast fatal_wait, so the watchdog would abort
    // with a misleading "request stayed in service". Fail stop now, naming the cause. The reader is closed (ready()
    // false), as after a failed reset below.
    if (fixed_reads())
      throw std::runtime_error(
          std::string("expert stream io_uring: the kernel refused the NOP drain of unconsumed reads (") +
          std::strerror(-refused) + "); re-registering the fixed buffers would exceed the watchdog's fatal_wait, so "
          "the reader is closed instead of resetting its ring");
    // Without registered buffers the reset is cheap (at most the fixed file table).
    // A failed reset leaves the reader closed (ready() false, so a later read() returns 0 and close() has nothing
    // to settle) and throws its reason to the caller: "io_uring ring reset failed", create_ring's own error, or
    // register_resources' refusal.
    bool injected = false;
    if constexpr (Build::kFaults) {
      injected = hooks_.reset_fail;
      hooks_.reset_fail = false;
    }
    if (injected || !create_ring()) throw std::runtime_error("io_uring ring reset failed");
    try {
      register_resources();
    } catch (...) {
      close_ring();
      throw;
    }
    diagnostics();
  }

  // Test only (fault word 30 bit 0, through FaultyReader): the next drain() that discards unconsumed SQEs finds its
  // NOP submission refused (EIO), as if the kernel had failed it. Fires once.
  void set_nop_flush_refused(bool refused)
    requires(Build::kFaults)
  {
    hooks_.nop_flush_refused = refused;
  }
  // Test only: publishes the prepared SQEs to the kernel's SQ tail without entering the kernel, as liburing's submit
  // does before an io_uring_enter that then fails. drain() must treat them as unconsumed and NOP them too.
  void publish_sq_without_enter()
    requires(Build::kFaults)
  {
    if (options_.sqpoll()) throw std::logic_error("publish_sq_without_enter needs a ring without SQPOLL");
    ring_.sq.sqe_head = ring_.sq.sqe_tail;
    __atomic_store_n(ring_.sq.ktail, ring_.sq.sqe_tail, __ATOMIC_RELEASE);
  }

  // Test only (fault word 30 bit 1, through FaultyReader): the next reset in drain() (a refused NOP drain outside
  // the fixed read modes) fails as if create_ring() had, after closing the old ring. Fires once.
  void set_ring_reset_fail(bool fail)
    requires(Build::kFaults)
  {
    hooks_.reset_fail = fail;
  }

 private:
  static bool soft_error(int rc) {
    return rc == -EINTR || rc == -EAGAIN || rc == -EBUSY;
  }
  [[noreturn]] static void error(const char* operation, int rc) {
    throw std::runtime_error(
        std::string("expert stream ") + operation + ": " + std::strerror(-rc) + " (" + std::to_string(rc) + ")");
  }
  static void spin_hint() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
  }
  void require_ready() const {
    if (!ready_) throw std::logic_error("io_uring reader is not ready");
    if ((options_.fixed_files || options_.read_mode != UringReadMode::Normal || options_.iopoll()) && !configured_)
      throw std::logic_error("io_uring resources must be configured before reading with these options");
  }
  bool create_ring() {
    io_uring_params p{};
    if (options_.sqpoll()) {
      p.flags |= IORING_SETUP_SQPOLL;
      p.sq_thread_idle = options_.sq_thread_idle_ms;
      if (options_.sq_thread_cpu >= 0) {
        p.flags |= IORING_SETUP_SQ_AFF;
        p.sq_thread_cpu = static_cast<unsigned>(options_.sq_thread_cpu);
      }
    }
    if (options_.iopoll()) p.flags |= IORING_SETUP_IOPOLL;
    // No SINGLE_ISSUER/DEFER_TASKRUN: opening, service, and temporary fill threads may take turns.
    requested_flags_ = p.flags;
    const int rc = io_uring_queue_init_params(depth_, &ring_, &p);
    if (rc < 0) {
      if (requested_flags_ != 0) error("requested io_uring setup mode is unsupported or unavailable", rc);
      // Explicit registration options refuse too, never quietly: the ring's own memory can be charged to the memlock
      // limit (RLIMIT_MEMLOCK=0 refuses it with ENOMEM on 6.12), the same limit registration is charged to.
      if (fixed_reads() || options_.fixed_files) {
        const std::string operation = "creating the ring for registered buffers or files (RLIMIT_MEMLOCK=" +
                                      memlock_limit() + ")";
        error(operation.c_str(), rc);
      }
      return false;
    }
    ready_ = true;
    params_ = p;
#if SGLANG_URING_HAS_READV_FIXED
    if (options_.read_mode == UringReadMode::ReadvFixed) {
      io_uring_probe* probe = io_uring_get_probe_ring(&ring_);
      if (probe == nullptr) {
        const int saved = errno;
        close_ring();
        error("probing READV_FIXED support", saved ? -saved : -EOPNOTSUPP);
      }
      const bool supported = io_uring_opcode_supported(probe, IORING_OP_READV_FIXED);
      io_uring_free_probe(probe);
      if (!supported) {
        close_ring();
        error("READ_MODE=readv_fixed is unsupported by the running kernel", -EOPNOTSUPP);
      }
    }
#endif
    return true;
  }
  // Registers the fixed files, then every region as row-aligned chunks in a sparse buffer table (also on drain()'s
  // fallback ring reset, taken only outside the fixed read modes when the kernel refuses the NOP drain). Any failure
  // is an explicit error (refuse), never a fallback to unregistered reads.
  void register_resources() {
    ++registrations_;
    if (!files_.empty()) {
      const int rc = io_uring_register_files(&ring_, files_.data(), static_cast<unsigned>(files_.size()));
      if (rc < 0) error("registering fixed files", rc);
      files_registered_ = true;
    }
    if (!regions_.empty()) {
      size_t slots = 0;
      uint64_t largest = 0;
      try {
        for (const auto& r : regions_) {
          for (const auto& chunk : sglang::io::plan_chunks(
                   reinterpret_cast<uint64_t>(r.base), r.bytes, r.row_bytes, chunk_cap_)) {  // throws: row > cap
            ++slots;
            largest = std::max(largest, chunk.length);
          }
        }
      } catch (const std::invalid_argument& e) {
        refuse(slots, largest, e.what());
      }
      if (slots > sglang::io::kMaxRegisteredBufferSlots)
        refuse(slots, largest, "more chunks than the kernel's 16384 slots");
      const auto t0 = std::chrono::steady_clock::now();
      if (!table_.init(&ring_, static_cast<unsigned>(slots)))
        refuse(slots, largest, "sparse buffer table unsupported: " + table_.last_error_context());
      for (const auto& r : regions_) {
        if (table_.add(reinterpret_cast<uint64_t>(r.base), r.bytes, r.row_bytes, chunk_cap_) == 0)
          refuse(slots, largest, table_.last_error_context());
      }
      register_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      buffers_registered_ = true;
    }
  }
  // Tears down whatever registration exists, then throws naming the registration and the memlock limit.
  [[noreturn]] void refuse(size_t chunks, uint64_t largest, const std::string& why) {
    if (table_.supported()) io_uring_unregister_buffers(&ring_);
    table_.clear();
    buffers_registered_ = false;
    if (files_registered_) io_uring_unregister_files(&ring_);
    files_registered_ = false;
    uint64_t bytes = 0;
    for (const auto& r : regions_)
      bytes += r.bytes;
    const std::string memlock = memlock_limit();
    throw std::runtime_error(
        "expert stream registering fixed buffers (regions=" + std::to_string(regions_.size()) +
        ", chunks=" + std::to_string(chunks) + ", bytes=" + std::to_string(bytes) + ", largest=" +
        std::to_string(largest) + ", cap=" + std::to_string(chunk_cap_) + ", RLIMIT_MEMLOCK=" + memlock + "): " + why);
  }
  static std::string memlock_limit() {
    rlimit limit{};
    if (getrlimit(RLIMIT_MEMLOCK, &limit) != 0) return "unknown";
    return limit.rlim_cur == RLIM_INFINITY ? std::string("unlimited") : std::to_string(limit.rlim_cur);
  }
  void close_ring() noexcept {
    if (!ready_) return;
    // Every request has retired (or, without SQPOLL, never entered the kernel).
    if (buffers_registered_) {
      io_uring_unregister_buffers(&ring_);
      table_.clear();
    }
    if (files_registered_) io_uring_unregister_files(&ring_);
    io_uring_queue_exit(&ring_);
    buffers_registered_ = files_registered_ = ready_ = false;
  }
  int mapped_fd(int fd) const {
    if (!options_.fixed_files) return fd;
    if (fd < 0 || static_cast<size_t>(fd) >= fd_index_.size() || fd_index_[static_cast<size_t>(fd)] < 0)
      throw std::invalid_argument("read fd is absent from the registered file table");
    return fd_index_[static_cast<size_t>(fd)];
  }
  void report_fixed() const
    requires Build::kMetrics
  {
    std::fprintf(
        stderr,
        "expert stream io_uring: fixed_reads=%llu fixed_cuts=%llu fanout_sqes=%llu\n",
        static_cast<unsigned long long>(fanout_.reads),
        static_cast<unsigned long long>(fanout_.cuts),
        static_cast<unsigned long long>(fanout_.sqes));
  }
  void finish_prep(io_uring_sqe* sqe, uint64_t tag) {
    if (options_.fixed_files) sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, tag);
    ++outstanding_;
  }
  void retire(io_uring_cqe* cqe) {
    io_uring_cqe_seen(&ring_, cqe);
    --outstanding_;
  }
  void wait_one() {
    io_uring_cqe* cqe = nullptr;
    if (options_.polls_in_wait()) {
      // Never io_uring_wait_cqe here: see UringOptions::polls_in_wait.
      for (;;) {
        const int rc = io_uring_peek_cqe(&ring_, &cqe);
        if (rc == 0) break;
        if (rc != -EAGAIN && rc != -EINTR) std::terminate();
        const int got = io_uring_get_events(&ring_);
        if (got < 0 && !soft_error(got)) std::terminate();
        spin_hint();
      }
      retire(cqe);
      return;
    }
    int rc;
    do {
      rc = io_uring_wait_cqe(&ring_, &cqe);
    } while (soft_error(rc));
    if (rc < 0) std::terminate();
    retire(cqe);
  }
  // Rewrites the last `n` prepared SQEs, none consumed by the kernel, as NOPs, then submits and retires them. Without
  // SQPOLL the kernel reads an SQE only inside io_uring_enter, so positions [sqe_tail - n, sqe_tail) still belong to
  // this thread, whether or not an earlier submit flushed them to the kernel's tail. create_ring never sets SQE128,
  // so an SQE's slot is its position. The whole SQE is zeroed: liburing's prep_* helpers leave flags
  // (IOSQE_FIXED_FILE) and buf_index to get_sqe. Returns 0, or the refusing submit's negative errno: every NOP the
  // kernel consumed has then retired, and the rest stay unconsumed and counted in outstanding_.
  int flush_as_nops(unsigned n) {
    const unsigned tail = ring_.sq.sqe_tail;
    for (unsigned i = tail - n; i != tail; ++i) {
      io_uring_sqe* sqe = &ring_.sq.sqes[i & ring_.sq.ring_mask];
      std::memset(static_cast<void*>(sqe), 0, sizeof(*sqe));
      io_uring_prep_nop(sqe);
      io_uring_sqe_set_data64(sqe, kNopTag);
    }
    int refused = 0;
    if constexpr (Build::kFaults) {
      refused = hooks_.nop_flush_refused ? -EIO : 0;
      hooks_.nop_flush_refused = false;
    }
    // EAGAIN (no request memory), EBUSY and EINTR are transient: retry with a backoff for up to kNopRetryWindow, well
    // under fatal_wait, before refusing (a refusal fail-stops a fixed read mode).
    const auto give_up = std::chrono::steady_clock::now() + kNopRetryWindow;
    auto backoff = std::chrono::microseconds(10);
    while (refused == 0 && io_uring_sq_ready(&ring_) != 0) {
      const int rc = io_uring_submit(&ring_);
      if (rc >= 0) continue;
      if (!soft_error(rc) || std::chrono::steady_clock::now() >= give_up) {
        refused = rc;
      } else {
        std::this_thread::sleep_for(backoff);
        backoff = std::min(backoff * 2, std::chrono::microseconds(10000));
      }
    }
    const unsigned unconsumed = io_uring_sq_ready(&ring_);
    while (outstanding_ > unconsumed)
      wait_one();  // a NOP completes at issue; IOPOLL rings retire it through the same reap loop
    return refused;
  }
  void diagnostics() const {
    if (!options_.diagnostics) return;
    std::fprintf(
        stderr,
        "expert stream io_uring: mode=%s read_mode=%s wait=%s requested_flags=0x%x effective_flags=0x%x "
        "requested_depth=%u sq_entries=%u cq_entries=%u features=0x%x fixed_files=%zu "
        "fixed_buffers=%zu registered_bytes=%llu sq_thread_idle_ms=%u sq_thread_cpu=%d "
        "regions=%zu chunks=%zu largest_chunk=%llu chunk_cap=%zu register_ms=%.1f read_cuts=%s:%d effective_wait=%s\n",
        options_.mode_name(),
        options_.read_mode_name(),
        options_.wait_mode == UringWaitMode::Spin ? "spin" : "block",
        requested_flags_,
        params_.flags,
        depth_,
        params_.sq_entries,
        params_.cq_entries,
        params_.features,
        files_registered_ ? files_.size() : 0,
        buffers_registered_ ? table_.chunks() : 0,
        static_cast<unsigned long long>(buffers_registered_ ? table_.bytes() : 0),
        options_.sq_thread_idle_ms,
        options_.sq_thread_cpu,
        regions_.size(),
        table_.chunks(),
        static_cast<unsigned long long>(table_.largest()),
        static_cast<size_t>(chunk_cap_),
        register_ms_,
        options_.read_cuts_name(),
        options_.read_cuts_on() ? 1 : 0,
        options_.effective_wait_name());
  }

  io_uring ring_{};
  io_uring_params params_{};
  UringOptions options_{};
  unsigned depth_ = 0;
  unsigned requested_flags_ = 0;
  unsigned outstanding_ = 0;
  bool ready_ = false;
  bool configured_ = false;
  bool files_registered_ = false;
  bool buffers_registered_ = false;
  std::vector<int> files_;
  std::vector<int> fd_index_;  // fd -> index in files_ (-1: absent); fixed files only
  sglang::io::RegisteredBufferTable table_;
  std::vector<RegisteredRegion> regions_;  // fixed read modes only
  size_t chunk_cap_ = sglang::io::kMaxRegisteredBufferBytes;
  // note_fanout()'s counters (spec M10): InstrBuild only.
  struct FanoutStats {
    uint64_t reads = 0, cuts = 0, sqes = 0, next_report = uint64_t{1} << 16;
  };
  struct NoFanoutStats {};
  [[no_unique_address]] std::conditional_t<Build::kMetrics, FanoutStats, NoFanoutStats> fanout_;
  double register_ms_ = 0;
  uint64_t registrations_ = 0;  // register_resources() calls (registrations())
  struct TestHooks {
    bool reset_fail = false;         // set_ring_reset_fail
    bool nop_flush_refused = false;  // set_nop_flush_refused
  };
  struct NoTestHooks {};
  [[no_unique_address]] std::conditional_t<Build::kFaults, TestHooks, NoTestHooks> hooks_;  // InstrBuild only
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(fanout_)>, "ProdBuild has no metrics");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(hooks_)>, "ProdBuild has no faults");
  // A discarded SQE's user_data (drain retires it unread). Not ~0: that is liburing's LIBURING_UDATA_TIMEOUT, whose
  // CQEs its peek swallows on kernels without IORING_FEAT_EXT_ARG.
  static constexpr uint64_t kNopTag = ~uint64_t{0} - 1;
  static constexpr std::chrono::milliseconds kNopRetryWindow{2000};  // soft errors on the NOP submit, then refuse
};

using UringReader = BasicUringReader<ProdBuild>;
using InstrUringReader = BasicUringReader<InstrBuild>;
static_assert(AsyncFileReader<UringReader>);
static_assert(AsyncFileReader<InstrUringReader>);
template <class R>
concept HasRingFaultHooks = requires(R& r) { r.set_nop_flush_refused(true); };
static_assert(!HasRingFaultHooks<UringReader>, "the production reader has no fault hooks");
static_assert(HasRingFaultHooks<InstrUringReader>);

}  // namespace sglang::expert_stream
