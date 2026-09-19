// Option C RAM-miss service for EXL3 streamed experts (DSV41 Phase 3b plan, D8-D19).
//
// This file grows in three plan tasks: the row reader (Task 10: io_uring superset
// reads into a page-aligned bounce, then Exl3ShardRowSource's per-name split into the
// pinned slabs), the C++-owned slot bookkeeping and request service (Task 11), and
// the service thread with its watchdog (Task 12). Nothing here makes a CUDA call:
// every write is a CPU store into (pinned) host memory (plan D9).

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <fcntl.h>
#include <immintrin.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sglang {
namespace exl3_ram_miss {

using tvm::ffi::TensorView;

constexpr int kBounceRows = 8;
constexpr unsigned kQueueDepth = 16;
constexpr int64_t kPage = 4096;

inline int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

struct Segment {
  int64_t name;
  int64_t dst;
  int64_t src;
  int64_t bytes;
};

struct Read {
  int64_t file;
  int64_t offset;
  int64_t length;
  int64_t start;
};

struct Tables {
  int64_t layers = 0;
  int64_t experts = 0;
  int64_t slot_bytes = 0;
  std::vector<std::string> paths;
  std::vector<int64_t> file_sizes;
  std::vector<Read> reads;
  std::vector<Segment> segments;
  std::vector<std::vector<uint8_t*>> slabs;
  std::vector<int64_t> row_bytes;
};

inline std::vector<int32_t> ids_of(TensorView tensor) {
  const auto* data = static_cast<const int64_t*>(tensor.data_ptr());
  return std::vector<int32_t>(data, data + tensor.size(0));
}

inline Tables tables_from(
    TensorView reads,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    const std::string& paths,
    int64_t slot_bytes) {
  Tables t;
  t.layers = reads.size(0);
  t.experts = reads.size(1);
  t.slot_bytes = slot_bytes;
  size_t start = 0;
  while (true) {
    const size_t end = paths.find('\n', start);
    t.paths.push_back(paths.substr(start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  const auto* sizes = static_cast<const int64_t*>(file_sizes.data_ptr());
  t.file_sizes.assign(sizes, sizes + file_sizes.size(0));
  const auto* read_data = static_cast<const int64_t*>(reads.data_ptr());
  t.reads.resize(static_cast<size_t>(t.layers * t.experts));
  for (size_t i = 0; i < t.reads.size(); ++i) {
    t.reads[i] = Read{read_data[4 * i], read_data[4 * i + 1], read_data[4 * i + 2], read_data[4 * i + 3]};
  }
  const auto* segment_data = static_cast<const int64_t*>(segments.data_ptr());
  t.segments.resize(static_cast<size_t>(segments.size(0)));
  for (size_t i = 0; i < t.segments.size(); ++i) {
    t.segments[i] = Segment{segment_data[4 * i], segment_data[4 * i + 1], segment_data[4 * i + 2], segment_data[4 * i + 3]};
  }
  const auto* slab_data = static_cast<const int64_t*>(slabs.data_ptr());
  const int64_t names = slabs.size(1);
  t.slabs.resize(static_cast<size_t>(t.layers));
  for (int64_t row = 0; row < t.layers; ++row) {
    for (int64_t name = 0; name < names; ++name) {
      t.slabs[row].push_back(reinterpret_cast<uint8_t*>(static_cast<intptr_t>(slab_data[row * names + name])));
    }
  }
  const auto* rows = static_cast<const int64_t*>(row_bytes.data_ptr());
  t.row_bytes.assign(rows, rows + row_bytes.size(0));
  return t;
}

// Test-only fault injection for RowReader (exl3_ram_miss_read_rows_faulted).
struct ReadFault {
  int submit_error = 0;       // errno the `submit_call`-th submit returns (0: no fault)
  int64_t submit_call = 0;    // 1-based count of submit-and-wait calls over the reader's life
  bool submit_first = false;  // submit the prepared SQEs before failing (reads are in flight)
  int cqe_error = 0;          // errno that replaces the `cqe_call`-th completion's result
  int64_t cqe_call = 0;       // 1-based count of reaped completions over the reader's life
};

// io_uring superset reads of whole expert rows into a page-aligned bounce, then the
// per-name split into the pinned slabs (Exl3ShardRowSource.read's copies).
class RowReader {
 public:
  RowReader(Tables tables, bool direct) : t_(std::move(tables)), direct_(direct) {}
  RowReader(const RowReader&) = delete;  // owns fds, the ring and the bounce
  RowReader& operator=(const RowReader&) = delete;

  ~RowReader() {
    if (ring_ready_) io_uring_queue_exit(&ring_);
    for (int fd : fds_) ::close(fd);
    std::free(bounce_);
  }

  const Tables& tables() const { return t_; }

  void set_fault(const ReadFault& fault) { fault_ = fault; }

  bool open() {
    for (const auto& path : t_.paths) {
      const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct_ ? O_DIRECT : 0));
      if (fd < 0) {
        std::fprintf(stderr, "ERROR exl3 RAM miss: open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
      }
      fds_.push_back(fd);
    }
    if (posix_memalign(reinterpret_cast<void**>(&bounce_), kPage, static_cast<size_t>(kBounceRows * t_.slot_bytes)) != 0) {
      bounce_ = nullptr;
      return false;
    }
    if (io_uring_queue_init(kQueueDepth, &ring_, 0) != 0) return false;
    ring_ready_ = true;
    return true;
  }

  // Read `experts` of streamed row `row` into `slots`, `step` rows per io_uring batch
  // (at most kBounceRows). `abandon()` runs before each batch; true stops the read.
  // Returns 1 when every row landed, 0 on an I/O error or short file, -1 when abandoned.
  // Every return leaves the ring empty: nothing in flight, nothing prepared (I1).
  //
  // This is the hot path and checks nothing: `row`, `experts` and `slots` must be in
  // range and `experts.size() == slots.size()`. The service (Task 11) and
  // read_rows_once (Python) validate at their boundaries.
  int read(
      int64_t row,
      const std::vector<int32_t>& experts,
      const std::vector<int64_t>& slots,
      size_t step,
      const std::function<bool()>& abandon) {
    if (!ring_ready_) return 0;
    step = std::max<size_t>(1, std::min<size_t>(step, kBounceRows));
    for (size_t first = 0; first < experts.size(); first += step) {
      if (abandon()) return -1;
      const size_t count = std::min<size_t>(step, experts.size() - first);
      std::vector<int64_t> done(count, 0);
      std::vector<int64_t> expected(count, 0);
      std::vector<int> retries(count, 0);
      std::vector<const Read*> reads(count);
      for (size_t i = 0; i < count; ++i) {
        reads[i] = &t_.reads[static_cast<size_t>(row * t_.experts + experts[first + i])];
        expected[i] = std::min(reads[i]->length, t_.file_sizes[reads[i]->file] - reads[i]->offset);
      }
      // SQEs prepared and not yet reaped (in the SQ ring or in the kernel). At most
      // kBounceRows < kQueueDepth, so io_uring_get_sqe never runs out.
      unsigned pending = 0;
      auto submit = [&](size_t i) {
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        const int64_t remaining = reads[i]->length - done[i];
        io_uring_prep_read(
            sqe, fds_[reads[i]->file], bounce_ + i * t_.slot_bytes + done[i], static_cast<unsigned>(remaining),
            static_cast<uint64_t>(reads[i]->offset + done[i]));
        io_uring_sqe_set_data64(sqe, i);
        ++pending;
      };
      for (size_t i = 0; i < count; ++i) submit(i);
      bool failed = false;
      int soft_errors = 0;
      while (pending > 0) {
        const int rc = submit_and_wait();
        if (rc < 0) {
          // -EINTR/-EAGAIN/-EBUSY: reap what has completed and submit again
          // (uring_file_reader.cpp). Anything else, or a soft error that never clears, fails.
          const bool soft = rc == -EINTR || rc == -EAGAIN || rc == -EBUSY;
          if (!soft || ++soft_errors > kMaxSoftErrors) {
            failed = true;
            break;
          }
        } else {
          soft_errors = 0;
        }
        io_uring_cqe* cqe;
        unsigned head;
        unsigned seen = 0;
        std::vector<size_t> again;
        io_uring_for_each_cqe(&ring_, head, cqe) {
          ++seen;
          --pending;
          const size_t i = static_cast<size_t>(io_uring_cqe_get_data64(cqe));
          int res = cqe->res;
          ++cqes_;
          if (fault_.cqe_error != 0 && cqes_ == fault_.cqe_call) res = -fault_.cqe_error;
          if (res == -EINTR || res == -EAGAIN) {
            if (++retries[i] > kMaxRetries) {
              failed = true;
            } else {
              again.push_back(i);  // resubmit the same range (M3)
            }
            continue;
          }
          if (res < 0 || (res == 0 && done[i] < expected[i])) {
            failed = true;
            continue;
          }
          done[i] += res;
          // A mid-file O_DIRECT read ends short only on a logical-block boundary, so
          // offset + done, bounce + done and length - done stay block-aligned and the
          // resubmit is a legal direct read. At EOF, done == expected: no resubmit.
          if (done[i] < expected[i]) again.push_back(i);
        }
        io_uring_cq_advance(&ring_, seen);
        if (failed) break;
        for (size_t i : again) submit(i);
      }
      if (failed) {
        drain(pending);
        return 0;
      }
      for (size_t i = 0; i < count; ++i) {
        const uint8_t* base = bounce_ + i * t_.slot_bytes + reads[i]->start;
        const int64_t slot = slots[first + i];
        for (const Segment& segment : t_.segments) {
          std::memcpy(
              t_.slabs[row][segment.name] + slot * t_.row_bytes[segment.name] + segment.dst, base + segment.src,
              static_cast<size_t>(segment.bytes));
        }
      }
    }
    return 1;
  }

 private:
  static constexpr int kMaxSoftErrors = 1000;
  static constexpr int kMaxRetries = 8;

  int submit_and_wait() {
    ++submits_;
    if (fault_.submit_error != 0 && submits_ == fault_.submit_call) {
      if (fault_.submit_first) io_uring_submit(&ring_);
      return -fault_.submit_error;
    }
    return io_uring_submit_and_wait(&ring_, 1);
  }

  // After a failure, empty the ring before the bounce is reused or freed: reap every
  // read the kernel holds, then drop SQEs that were prepared but never consumed by
  // resetting the ring (the kernel has not seen them, so nothing can write the bounce).
  // `pending` counts both; io_uring_sq_ready counts the unconsumed ones
  // (uring_file_reader.cpp abandon_after_submit_failure_).
  void drain(unsigned pending) {
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
      ring_ready_ = io_uring_queue_init(kQueueDepth, &ring_, 0) == 0;
      if (!ring_ready_) std::fprintf(stderr, "ERROR exl3 RAM miss: io_uring ring reset failed\n");
    }
  }

  Tables t_;
  bool direct_;
  std::vector<int> fds_;
  uint8_t* bounce_ = nullptr;
  io_uring ring_{};
  bool ring_ready_ = false;
  ReadFault fault_{};
  int64_t submits_ = 0;
  int64_t cqes_ = 0;
};

}  // namespace exl3_ram_miss

using exl3_ram_miss::TensorView;

namespace {

std::vector<int64_t> slots_of(TensorView slots) {
  const auto* data = static_cast<const int64_t*>(slots.data_ptr());
  return std::vector<int64_t>(data, data + slots.size(0));
}

}  // namespace

// Read `experts` of streamed row `row` into `slots` once, synchronously (tests, tools).
// Arguments are validated by the Python wrapper (read_rows_once).
int64_t exl3_ram_miss_read_rows(
    TensorView reads,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    int64_t slot_bytes,
    int64_t direct,
    int64_t row,
    TensorView experts,
    TensorView slots,
    int64_t step) {
  using namespace exl3_ram_miss;
  RowReader reader(tables_from(reads, file_sizes, segments, slabs, row_bytes, paths, slot_bytes), direct != 0);
  if (!reader.open()) return 0;
  return reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), [] { return false; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows, exl3_ram_miss_read_rows);

// Test only: one reader reads `experts` into `slots` with `fault` injected
// ([submit_error, submit_call, submit_first, cqe_error, cqe_call]), then reads
// `then_experts` into `then_slots` with no fault. Results go to `results[0..1]`.
void exl3_ram_miss_read_rows_faulted(
    TensorView reads,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    int64_t slot_bytes,
    int64_t direct,
    int64_t row,
    TensorView experts,
    TensorView slots,
    TensorView then_experts,
    TensorView then_slots,
    TensorView fault,
    TensorView results) {
  using namespace exl3_ram_miss;
  auto* out = static_cast<int64_t*>(results.data_ptr());
  RowReader reader(tables_from(reads, file_sizes, segments, slabs, row_bytes, paths, slot_bytes), direct != 0);
  if (!reader.open()) {
    out[0] = out[1] = 0;
    return;
  }
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  reader.set_fault(ReadFault{static_cast<int>(f[0]), f[1], f[2] != 0, static_cast<int>(f[3]), f[4]});
  const auto never = [] { return false; };
  out[0] = reader.read(row, ids_of(experts), slots_of(slots), kBounceRows, never);
  reader.set_fault(ReadFault{});
  out[1] = reader.read(row, ids_of(then_experts), slots_of(then_slots), kBounceRows, never);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_faulted, exl3_ram_miss_read_rows_faulted);

}  // namespace sglang
