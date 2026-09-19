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

// io_uring superset reads of whole expert rows into a page-aligned bounce, then the
// per-name split into the pinned slabs (Exl3ShardRowSource.read's copies).
class RowReader {
 public:
  RowReader(Tables tables, bool direct) : t_(std::move(tables)), direct_(direct) {}

  ~RowReader() {
    if (ring_ready_) io_uring_queue_exit(&ring_);
    for (int fd : fds_) ::close(fd);
    std::free(bounce_);
  }

  const Tables& tables() const { return t_; }

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
  int read(
      int64_t row,
      const std::vector<int32_t>& experts,
      const std::vector<int64_t>& slots,
      size_t step,
      const std::function<bool()>& abandon) {
    step = std::max<size_t>(1, std::min<size_t>(step, kBounceRows));
    for (size_t first = 0; first < experts.size(); first += step) {
      if (abandon()) return -1;
      const size_t count = std::min<size_t>(step, experts.size() - first);
      std::vector<int64_t> done(count, 0);
      std::vector<int64_t> expected(count, 0);
      std::vector<const Read*> reads(count);
      for (size_t i = 0; i < count; ++i) {
        reads[i] = &t_.reads[static_cast<size_t>(row * t_.experts + experts[first + i])];
        expected[i] = std::min(reads[i]->length, t_.file_sizes[reads[i]->file] - reads[i]->offset);
      }
      size_t pending = 0;
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
      while (pending > 0) {
        if (io_uring_submit_and_wait(&ring_, 1) < 0) return 0;
        io_uring_cqe* cqe;
        unsigned head;
        unsigned seen = 0;
        std::vector<size_t> again;
        bool failed = false;
        io_uring_for_each_cqe(&ring_, head, cqe) {
          ++seen;
          --pending;
          const size_t i = static_cast<size_t>(io_uring_cqe_get_data64(cqe));
          if (cqe->res < 0 || (cqe->res == 0 && done[i] < expected[i])) {
            failed = true;
            continue;
          }
          done[i] += cqe->res;
          if (done[i] < expected[i]) again.push_back(i);
        }
        io_uring_cq_advance(&ring_, seen);
        if (failed) {
          while (pending > 0) {  // drain what is still in flight before the bounce is reused
            io_uring_cqe* rest;
            if (io_uring_wait_cqe(&ring_, &rest) == 0) io_uring_cqe_seen(&ring_, rest);
            --pending;
          }
          return 0;
        }
        for (size_t i : again) submit(i);
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
  Tables t_;
  bool direct_;
  std::vector<int> fds_;
  uint8_t* bounce_ = nullptr;
  io_uring ring_{};
  bool ring_ready_ = false;
};

}  // namespace exl3_ram_miss

using exl3_ram_miss::TensorView;

// Read `experts` of streamed row `row` into `slots` once, synchronously (tests, tools).
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
    TensorView slots) {
  using namespace exl3_ram_miss;
  RowReader reader(tables_from(reads, file_sizes, segments, slabs, row_bytes, paths, slot_bytes), direct != 0);
  if (!reader.open()) return 0;
  const auto* slot_data = static_cast<const int64_t*>(slots.data_ptr());
  return reader.read(
      row, ids_of(experts), std::vector<int64_t>(slot_data, slot_data + slots.size(0)), kBounceRows,
      [] { return false; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows, exl3_ram_miss_read_rows);

}  // namespace sglang
