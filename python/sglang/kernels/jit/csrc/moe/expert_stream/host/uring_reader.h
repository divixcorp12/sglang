// AsyncFileReader over one io_uring ring.
#pragma once

#include <liburing.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <exception>

#include "file_reader.h"

namespace sglang {
namespace expert_stream {

class UringReader {
 public:
  UringReader() = default;
  UringReader(const UringReader&) = delete;
  UringReader& operator=(const UringReader&) = delete;
  ~UringReader() {
    if (ready_) io_uring_queue_exit(&ring_);
  }

  // Flags 0, deliberately: SINGLE_ISSUER / DEFER_TASKRUN would bind the ring to the opening thread (see file_reader.h).
  bool init(unsigned depth) {
    depth_ = depth;
    ready_ = io_uring_queue_init(depth, &ring_, 0) == 0;
    return ready_;
  }
  bool ready() const { return ready_; }

  bool prep_read(int fd, void* buf, unsigned len, uint64_t off, uint64_t tag) {
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    io_uring_prep_read(sqe, fd, buf, len, off);
    io_uring_sqe_set_data64(sqe, tag);
    return true;
  }
  bool prep_readv(int fd, const iovec* iov, unsigned count, uint64_t off, uint64_t tag) {
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) return false;
    io_uring_prep_readv(sqe, fd, iov, count, off);
    io_uring_sqe_set_data64(sqe, tag);
    return true;
  }

  // A submit that consumes nothing while nothing is in flight would make the wait below block in
  // GETEVENTS for a completion no in-kernel SQE can produce (the state submit_short_call imitates).
  // Guarding it costs a second syscall on every batch of the decode path, and the service watchdog
  // already aborts a read that stays in service, so this is left to the watchdog deliberately.
  // If it ever does surface, the signature is busy_since_ non-zero with pending > 0 and an empty
  // completion queue; the guard would be to pass wait_nr = 0 whenever io_uring_sq_ready() > 0.
  int submit(unsigned wait_nr) { return wait_nr != 0 ? io_uring_submit_and_wait(&ring_, wait_nr) : io_uring_submit(&ring_); }

  unsigned reap(std::vector<ReadCompletion>& out) {
    io_uring_cqe* cqe;
    unsigned head;
    unsigned seen = 0;
    io_uring_for_each_cqe(&ring_, head, cqe) {
      ++seen;
      out.push_back(ReadCompletion{io_uring_cqe_get_data64(cqe), cqe->res});
    }
    io_uring_cq_advance(&ring_, seen);
    return seen;
  }

  // After a failure, empty the ring before the bounce is reused or freed: reap every
  // read the kernel holds, then drop SQEs that were prepared but never consumed by
  // resetting the ring (the kernel has not seen them, so nothing can write the bounce).
  // `pending` counts both; io_uring_sq_ready counts the unconsumed ones
  // (uring_file_reader.cpp abandon_after_submit_failure_).
  void drain(unsigned pending);

 private:
  io_uring ring_{};
  unsigned depth_ = 0;
  bool ready_ = false;
};

inline void UringReader::drain(unsigned pending) {
  const unsigned unsubmitted = std::min(pending, io_uring_sq_ready(&ring_));
  unsigned in_kernel = pending - unsubmitted;
  while (in_kernel > 0) {
    io_uring_cqe* cqe = nullptr;
    const int rc = io_uring_wait_cqe(&ring_, &cqe);
    if (rc == -EINTR || rc == -EAGAIN) continue;
    // A read could still land in the bounce later: no safe way to go on.
    if (rc < 0) std::terminate();
    io_uring_cqe_seen(&ring_, cqe);
    --in_kernel;
  }
  if (unsubmitted > 0) {
    io_uring_queue_exit(&ring_);
    ready_ = io_uring_queue_init(depth_, &ring_, 0) == 0;
    if (!ready_) std::fprintf(stderr, "ERROR exl3 RAM miss: io_uring ring reset failed\n");
  }
}

static_assert(AsyncFileReader<UringReader>);

}  // namespace expert_stream
}  // namespace sglang
