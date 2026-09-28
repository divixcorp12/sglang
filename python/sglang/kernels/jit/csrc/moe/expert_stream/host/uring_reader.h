// AsyncFileReader over one io_uring ring shared by every model file.
#pragma once

#include "file_reader.h"
#include "uring_options.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <liburing.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#if defined(IO_URING_VERSION_MAJOR) && \
    (IO_URING_VERSION_MAJOR > 2 || (IO_URING_VERSION_MAJOR == 2 && IO_URING_VERSION_MINOR >= 10))
#define SGLANG_URING_HAS_READV_FIXED 1
#else
#define SGLANG_URING_HAS_READV_FIXED 0
#endif

namespace sglang::expert_stream {

class UringReader {
 public:
  UringReader() = default;
  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;
  ~UringReader() {
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

  // Buffers describe their complete live allocations, never a guessed interval between unrelated slabs.
  // Registration is independent of CUDA pinning. The owner must retain these allocations through close().
  void configure_resources(const std::vector<int>& fds, const std::vector<iovec>& buffers, bool direct) {
    if (!ready_ || configured_ || outstanding_ != 0)
      throw std::logic_error("io_uring resources must be configured once on a ready, idle reader");
    if (options_.iopoll() && !direct)
      throw std::invalid_argument("SGLANG_EXPERT_STREAM_URING_MODE with iopoll requires O_DIRECT files");
    if (options_.fixed_files) {
      if (fds.empty()) throw std::invalid_argument("fixed files requested with an empty file table");
      for (int fd : fds)
        if (fd < 0) throw std::invalid_argument("fixed file table contains an invalid fd");
      files_ = fds;
    }
    if (options_.read_mode != UringReadMode::Normal) {
      if (buffers.empty() || buffers.size() > 65536)
        throw std::invalid_argument("fixed reads require between 1 and 65536 registered buffer regions");
      for (const iovec& b : buffers) {
        const auto start = reinterpret_cast<uintptr_t>(b.iov_base);
        if (start == 0 || b.iov_len == 0 || b.iov_len > std::numeric_limits<uintptr_t>::max() - start)
          throw std::invalid_argument("invalid registered buffer region");
      }
      buffers_ = buffers;
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
        std::terminate();  // Never let a caller free memory that the kernel might still own.
      }
      close_ring();
    }
    files_.clear();
    buffers_.clear();
    scalar_iovecs_.clear();
    configured_ = false;
  }

  bool prep_read(int fd, void* buf, unsigned len, uint64_t off, uint64_t tag) {
    require_ready();
    const int file = mapped_fd(fd);
    const iovec range{buf, len};
    const int buffer = options_.read_mode == UringReadMode::Normal ? -1 : buffer_index(&range, 1);
    // Node storage keeps the scalar iovec stable across later preparations and SQPOLL consumption.
    auto scalar = scalar_iovecs_.end();
    if (options_.read_mode == UringReadMode::ReadvFixed) {
      const auto inserted = scalar_iovecs_.emplace(tag, range);
      if (!inserted.second) throw std::invalid_argument("duplicate live scalar readv_fixed tag");
      scalar = inserted.first;
    }
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) {
      if (scalar != scalar_iovecs_.end()) scalar_iovecs_.erase(scalar);
      return false;
    }
    if (options_.read_mode == UringReadMode::Normal)
      io_uring_prep_read(sqe, file, buf, len, off);
    else if (options_.read_mode == UringReadMode::Fixed)
      io_uring_prep_read_fixed(sqe, file, buf, len, off, buffer);
#if SGLANG_URING_HAS_READV_FIXED
    else
      io_uring_prep_readv_fixed(sqe, file, &scalar->second, 1, off, 0, buffer);
#endif
    finish_prep(sqe, tag);
    return true;
  }

  bool prep_readv(int fd, const iovec* iov, unsigned count, uint64_t off, uint64_t tag) {
    require_ready();
    if (iov == nullptr || count == 0) throw std::invalid_argument("readv requires a nonempty iovec array");
    if (options_.read_mode == UringReadMode::Fixed) {
      if (count != 1)
        throw std::invalid_argument("READ_MODE=fixed supports one iovec; use readv_fixed for scattered reads");
      if (iov[0].iov_len > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("fixed read length exceeds unsigned range");
      return prep_read(fd, iov[0].iov_base, static_cast<unsigned>(iov[0].iov_len), off, tag);
    }
    const int file = mapped_fd(fd);
    const int buffer = options_.read_mode == UringReadMode::ReadvFixed ? buffer_index(iov, count) : -1;
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    if (options_.read_mode == UringReadMode::Normal) io_uring_prep_readv(sqe, file, iov, count, off);
#if SGLANG_URING_HAS_READV_FIXED
    else
      io_uring_prep_readv_fixed(sqe, file, iov, count, off, 0, buffer);
#endif
    finish_prep(sqe, tag);
    return true;
  }

  int submit(unsigned wait_nr) {
    if (options_.wait_mode == UringWaitMode::Block)
      return wait_nr != 0 ? io_uring_submit_and_wait(&ring_, wait_nr) : io_uring_submit(&ring_);
    const int submitted = io_uring_submit(&ring_);
    if (submitted < 0 || wait_nr == 0) return submitted;
    while (io_uring_cq_ready(&ring_) < wait_nr) {
      // Short submissions must be retried: waiting on an SQE still in userspace cannot make progress.
      if (io_uring_sq_ready(&ring_) != 0) {
        const int rc = io_uring_submit(&ring_);
        if (rc < 0) return rc;
      }
      // IOPOLL needs GETEVENTS to poll storage. SQPOLL alone can observe its CQ entirely in userspace.
      if (options_.iopoll()) {
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
      const uint64_t tag = io_uring_cqe_get_data64(cqe);
      out.push_back(ReadCompletion{tag, cqe->res});
      scalar_iovecs_.erase(tag);
      ++seen;
    }
    io_uring_cq_advance(&ring_, seen);
    if (seen > outstanding_) std::terminate();
    outstanding_ -= seen;
    return seen;
  }

  void drain(unsigned pending) {
    if (pending != outstanding_) std::terminate();
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
    if (unsubmitted != 0) {
      close_ring();
      outstanding_ = 0;
      scalar_iovecs_.clear();
      if (!create_ring()) throw std::runtime_error("io_uring ring reset failed");
      register_resources();
      diagnostics();
    }
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
  void register_resources() {
    if (!files_.empty()) {
      const int rc = io_uring_register_files(&ring_, files_.data(), static_cast<unsigned>(files_.size()));
      if (rc < 0) error("registering fixed files", rc);
      files_registered_ = true;
    }
    if (!buffers_.empty()) {
      const int rc = io_uring_register_buffers(&ring_, buffers_.data(), static_cast<unsigned>(buffers_.size()));
      if (rc < 0) {
        size_t total = 0, largest = 0;
        for (const auto& b : buffers_) {
          total += b.iov_len;
          largest = std::max(largest, b.iov_len);
        }
        const std::string operation = "registering fixed buffers (regions=" + std::to_string(buffers_.size()) +
                                      ", bytes=" + std::to_string(total) + ", largest=" + std::to_string(largest) +
                                      "; check RLIMIT_MEMLOCK and kernel per-buffer limits, commonly 1 GiB)";
        error(operation.c_str(), rc);
      }
      buffers_registered_ = true;
    }
  }
  void close_ring() noexcept {
    if (!ready_) return;
    // Every request has retired (or, without SQPOLL, never entered the kernel).
    if (buffers_registered_) io_uring_unregister_buffers(&ring_);
    if (files_registered_) io_uring_unregister_files(&ring_);
    io_uring_queue_exit(&ring_);
    buffers_registered_ = files_registered_ = ready_ = false;
  }
  int mapped_fd(int fd) const {
    if (!options_.fixed_files) return fd;
    const auto it = std::find(files_.begin(), files_.end(), fd);
    if (it == files_.end()) throw std::invalid_argument("read fd is absent from the registered file table");
    return static_cast<int>(it - files_.begin());
  }
  int buffer_index(const iovec* iov, unsigned count) const {
    for (size_t index = 0; index < buffers_.size(); ++index) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(buffers_[index].iov_base);
      const size_t length = buffers_[index].iov_len;
      bool fits = true;
      for (unsigned j = 0; j < count; ++j) {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(iov[j].iov_base);
        if (addr < base || addr - base > length || iov[j].iov_len > length - (addr - base)) {
          fits = false;
          break;
        }
      }
      if (fits) return static_cast<int>(index);
    }
    throw std::invalid_argument("fixed read ranges must all lie within ONE registered buffer region");
  }
  void finish_prep(io_uring_sqe* sqe, uint64_t tag) {
    if (options_.fixed_files) sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, tag);
    ++outstanding_;
  }
  void retire(io_uring_cqe* cqe) {
    scalar_iovecs_.erase(io_uring_cqe_get_data64(cqe));
    io_uring_cqe_seen(&ring_, cqe);
    --outstanding_;
  }
  void wait_one() {
    io_uring_cqe* cqe = nullptr;
    int rc;
    do {
      rc = io_uring_wait_cqe(&ring_, &cqe);
    } while (soft_error(rc));
    if (rc < 0) std::terminate();
    retire(cqe);
  }
  void diagnostics() const {
    if (!options_.diagnostics) return;
    size_t bytes = 0;
    for (const auto& b : buffers_)
      bytes += b.iov_len;
    std::fprintf(
        stderr,
        "expert stream io_uring: mode=%s read_mode=%s wait=%s requested_flags=0x%x effective_flags=0x%x "
        "requested_depth=%u sq_entries=%u cq_entries=%u features=0x%x fixed_files=%zu "
        "fixed_buffers=%zu registered_bytes=%zu sq_thread_idle_ms=%u sq_thread_cpu=%d\n",
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
        buffers_registered_ ? buffers_.size() : 0,
        buffers_registered_ ? bytes : 0,
        options_.sq_thread_idle_ms,
        options_.sq_thread_cpu);
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
  std::vector<iovec> buffers_;
  std::unordered_map<uint64_t, iovec> scalar_iovecs_;
};

static_assert(AsyncFileReader<UringReader>);

}  // namespace sglang::expert_stream
