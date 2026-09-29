// The asynchronous positional reader a RowReader drives: prepare, submit, reap, and settle what is in flight.
#pragma once

#include <sys/uio.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sglang {
namespace expert_stream {

// One reaped read: the tag it was prepared with (RowReader: generation << 32 | descriptor) and bytes or -errno.
struct ReadCompletion {
  uint64_t data;
  int res;
};

// One thread drives a reader at a time, but not always the thread that built it (the service opens on the Python
// thread and reads on its own), so an implementation must not bind itself to its creating thread.
// Buffers and iovec arrays passed to prep_* stay the caller's and must outlive the read's completion.
template <typename R>
concept AsyncFileReader = requires(
    R& r,
    const R& cr,
    int fd,
    void* buf,
    const iovec* iov,
    unsigned n,
    uint64_t off,
    uint64_t tag,
    std::vector<ReadCompletion>& out) {
  { r.init(n) } -> std::same_as<bool>;  // n: reads prepared and not yet reaped, at most
  { cr.ready() } -> std::same_as<bool>;
  { r.prep_read(fd, buf, n, off, tag) } -> std::same_as<bool>;  // false: no room, reap first
  { r.prep_readv(fd, iov, n, off, tag) } -> std::same_as<bool>;
  { r.submit(n) } -> std::same_as<int>;       // wait for n completions (0: none); <0 is -errno
  { r.reap(out) } -> std::same_as<unsigned>;  // appends, never blocks
  { r.drain(n) } -> std::same_as<void>;       // n prepared-or-in-flight reads: settle all
};

// Memory a reader registers with its ring: one named slab (or the bounce), `bytes` long, made of `row_bytes` rows. The
// registration is cut on row boundaries into chunks of at most 1 GiB (io::plan_chunks), so no read's iovec, which lies
// inside one row, ever straddles two registered buffers. Defined here, not in row_tables.h, so uring_reader.h stays
// free of the TVM headers the tables need.
struct RegisteredRegion {
  void* base;
  size_t bytes;
  size_t row_bytes;
};

// One leg of a fixed read: iovecs [first, first + count) of the read, all in registered buffer `buffer`, `bytes` long.
struct FixedLeg {
  unsigned first, count;
  int buffer;
  size_t bytes;
};

// Test-only description of a submit fault, shared by every AsyncFileReader decorator that injects one.
struct SubmitFault {
  int error = 0;              // errno the `call`-th submit returns (0: none)
  int64_t call = 0;           // 1-based count of submits over the reader's life
  bool submit_first = false;  // submit the prepared reads before failing (reads are then in flight)
  int64_t short_call = 0;     // that submit consumes nothing and reports success
  // The next ring reset (drain()'s fallback after a refused NOP drain) fails as if re-creating the ring had failed.
  // Readers without a ring reset ignore it.
  bool ring_reset_fail = false;
  // The next drain that discards unconsumed SQEs finds its NOP submission refused: a fixed read mode then throws, any
  // other resets the ring (which ring_reset_fail can then fail). Readers without a ring ignore it.
  bool nop_flush_refused = false;
};

}  // namespace expert_stream
}  // namespace sglang
