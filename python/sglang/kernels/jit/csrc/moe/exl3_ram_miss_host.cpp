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

// One part of a row's aligned read: `length` bytes of `file` at `offset`, into the row's bounce
// slot at `dest`. A row is `parts` extents (one per mirror root); a zero-length one reads nothing.
struct Read {
  int64_t file;
  int64_t offset;
  int64_t length;
  int64_t dest;
};

struct Tables {
  int64_t layers = 0;
  int64_t experts = 0;
  int64_t parts = 1;
  int64_t slot_bytes = 0;
  std::vector<std::string> paths;
  std::vector<int64_t> file_sizes;  // the SOURCE size of every file, mirrors included
  std::vector<Read> extents;        // [layers][experts][parts]
  std::vector<int64_t> starts;      // [layers][experts]: where the row starts in its aligned superset
  std::vector<Segment> segments;
  std::vector<std::vector<uint8_t*>> slabs;
  std::vector<int64_t> row_bytes;
};

inline std::vector<int32_t> ids_of(TensorView tensor) {
  const auto* data = static_cast<const int64_t*>(tensor.data_ptr());
  return std::vector<int32_t>(data, data + tensor.size(0));
}

inline Tables tables_from(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    const std::string& paths,
    int64_t slot_bytes) {
  Tables t;
  t.layers = extents.size(0);
  t.experts = extents.size(1);
  t.parts = extents.size(2);
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
  const auto* extent_data = static_cast<const int64_t*>(extents.data_ptr());
  t.extents.resize(static_cast<size_t>(t.layers * t.experts * t.parts));
  for (size_t i = 0; i < t.extents.size(); ++i) {
    t.extents[i] = Read{extent_data[4 * i], extent_data[4 * i + 1], extent_data[4 * i + 2], extent_data[4 * i + 3]};
    // The reader writes each extent into its row's bounce slot and reads its file without
    // checking again, so a table that would write outside the slot or name no file is refused here.
    const Read& e = t.extents[i];
    if (e.file < 0 || e.file >= static_cast<int64_t>(t.paths.size()) ||
        e.file >= static_cast<int64_t>(t.file_sizes.size()) || e.offset < 0 || e.length < 0 || e.dest < 0 ||
        e.dest + e.length > slot_bytes) {
      throw std::runtime_error("exl3 RAM miss: an extent names no file or falls outside its bounce slot");
    }
  }
  const auto* start_data = static_cast<const int64_t*>(starts.data_ptr());
  t.starts.assign(start_data, start_data + t.layers * t.experts);
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
  // Per-extent faults, keyed by the extent's part index (-1: none). They hit the FIRST completion
  // of a part-`part` extent, whichever row it is in and however the kernel orders completions.
  int64_t part = -1;
  int part_error = 0;         // errno that replaces that completion's result
  int64_t part_short = 0;     // >0: that completion reports at most this many bytes (block multiple)
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

  void set_fault(const ReadFault& fault) {
    fault_ = fault;
    part_fired_ = false;
  }

  // Completions reaped over the reader's life (tests: a zero-length extent must add none).
  int64_t cqes() const { return cqes_; }

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
    if (io_uring_queue_init(queue_depth(), &ring_, 0) != 0) return false;
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
      // One entry per extent, j = i * parts + p for part p of row i of this batch.
      const size_t parts = static_cast<size_t>(t_.parts);
      std::vector<int64_t> done(count * parts, 0);
      std::vector<int64_t> expected(count * parts, 0);
      std::vector<int> retries(count * parts, 0);
      std::vector<const Read*> reads(count * parts);
      for (size_t i = 0; i < count; ++i) {
        const size_t base = static_cast<size_t>(row * t_.experts + experts[first + i]) * parts;
        for (size_t p = 0; p < parts; ++p) {
          const Read* extent = &t_.extents[base + p];
          reads[i * parts + p] = extent;
          // Per extent, against the file that extent reads.
          expected[i * parts + p] = std::min(extent->length, t_.file_sizes[extent->file] - extent->offset);
        }
      }
      // SQEs prepared and not yet reaped (in the SQ ring or in the kernel). At most
      // kBounceRows * parts, and the ring holds kQueueDepth * parts, so io_uring_get_sqe never runs out.
      unsigned pending = 0;
      auto submit = [&](size_t j) {
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        const int64_t remaining = reads[j]->length - done[j];
        io_uring_prep_read(
            sqe, fds_[reads[j]->file], bounce_ + (j / parts) * t_.slot_bytes + reads[j]->dest + done[j],
            static_cast<unsigned>(remaining), static_cast<uint64_t>(reads[j]->offset + done[j]));
        io_uring_sqe_set_data64(sqe, j);
        ++pending;
      };
      // A zero-length extent is a root that serves none of this row: no read, not pending.
      for (size_t j = 0; j < count * parts; ++j) {
        if (reads[j]->length > 0) submit(j);
      }
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
        std::vector<size_t> again;  // extents to resubmit
        io_uring_for_each_cqe(&ring_, head, cqe) {
          ++seen;
          --pending;
          // The extent, not the row: two parts of one row complete independently.
          const size_t j = static_cast<size_t>(io_uring_cqe_get_data64(cqe));
          int res = cqe->res;
          ++cqes_;
          if (fault_.cqe_error != 0 && cqes_ == fault_.cqe_call) res = -fault_.cqe_error;
          if (fault_.part >= 0 && !part_fired_ && static_cast<int64_t>(j % parts) == fault_.part) {
            if (fault_.part_error != 0) {
              part_fired_ = true;
              res = -fault_.part_error;
            } else if (fault_.part_short > 0 && res > fault_.part_short) {
              part_fired_ = true;
              res = static_cast<int>(fault_.part_short);
            }
          }
          if (res == -EINTR || res == -EAGAIN) {
            if (++retries[j] > kMaxRetries) {
              failed = true;
            } else {
              again.push_back(j);  // resubmit the same range (M3)
            }
            continue;
          }
          if (res < 0 || (res == 0 && done[j] < expected[j])) {
            failed = true;
            continue;
          }
          done[j] += res;
          // A mid-file O_DIRECT read ends short only on a logical-block boundary, so
          // offset + done, bounce + dest + done and length - done stay block-aligned (an
          // extent's offset, dest and length are whole pages) and the resubmit is a legal
          // direct read of just this extent. At EOF, done == expected: no resubmit.
          if (done[j] < expected[j]) again.push_back(j);
        }
        io_uring_cq_advance(&ring_, seen);
        if (failed) break;
        for (size_t j : again) submit(j);
      }
      if (failed) {
        drain(pending);
        return 0;
      }
      for (size_t i = 0; i < count; ++i) {
        // The row's parts landed contiguously, so its segments split from one base.
        const uint8_t* base =
            bounce_ + i * t_.slot_bytes + t_.starts[static_cast<size_t>(row * t_.experts + experts[first + i])];
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

  // Room for every extent of a full batch: kBounceRows rows of `parts` extents.
  unsigned queue_depth() const { return kQueueDepth * static_cast<unsigned>(t_.parts); }

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
      ring_ready_ = io_uring_queue_init(queue_depth(), &ring_, 0) == 0;
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
  bool part_fired_ = false;
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
    TensorView extents,
    TensorView starts,
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
  RowReader reader(tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, slot_bytes), direct != 0);
  if (!reader.open()) return 0;
  return reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), [] { return false; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows, exl3_ram_miss_read_rows);

// Test only: one reader reads `experts` into `slots` with `fault` injected
// ([submit_error, submit_call, submit_first, cqe_error, cqe_call, part, part_error, part_short]),
// then reads `then_experts` into `then_slots` with no fault. Results go to `results[0..3]`: the two
// reads' results, then the completions the reader had reaped after each.
void exl3_ram_miss_read_rows_faulted(
    TensorView extents,
    TensorView starts,
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
  RowReader reader(tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, slot_bytes), direct != 0);
  if (!reader.open()) {
    out[0] = out[1] = out[2] = out[3] = 0;
    return;
  }
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  reader.set_fault(
      ReadFault{static_cast<int>(f[0]), f[1], f[2] != 0, static_cast<int>(f[3]), f[4], f[5], static_cast<int>(f[6]), f[7]});
  const auto never = [] { return false; };
  out[0] = reader.read(row, ids_of(experts), slots_of(slots), kBounceRows, never);
  out[2] = reader.cqes();
  reader.set_fault(ReadFault{});
  out[1] = reader.read(row, ids_of(then_experts), slots_of(then_slots), kBounceRows, never);
  out[3] = reader.cqes();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_faulted, exl3_ram_miss_read_rows_faulted);

namespace exl3_ram_miss {

// ---- Request page (plan D10) ----
constexpr int64_t kDemandHead = 0;
constexpr int64_t kDemandDone = 4;
constexpr int64_t kFatal = 8;
constexpr int64_t kAdviseHead = 16;
constexpr int64_t kAdviseDone = 20;
constexpr int64_t kBusySeq = 24;
constexpr int64_t kHeartbeat = 28;
constexpr int64_t kRecordBytes = 128;
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kAdviseRing = kDemandRing + kDemandRecords * kRecordBytes;
constexpr uint32_t kAdviseRecords = 64;
constexpr int kMaxIds = 8;
constexpr int64_t kRecSeq = 0;
constexpr int64_t kRecRow = 4;
constexpr int64_t kRecNeedCount = 6;
constexpr int64_t kRecProtectCount = 8;
constexpr int64_t kRecStatus = 10;
constexpr int64_t kRecAfter = 12;
constexpr int64_t kRecNeed = 16;
constexpr int64_t kRecProtect = 48;
// uint32: nonzero when the device waits on this demand record (need non-empty or advise on).
constexpr int64_t kRecArmed = 80;
constexpr uint16_t kServed = 1;
constexpr uint16_t kFailed = 2;

enum : uint8_t { kFree = 0, kLoading = 1, kReady = 2 };

enum Counter : int {
  kServedRequests = 0,
  kTouchOnly,
  kRowsRead,
  kReadErrors,
  kEvictions,
  kOverruns,
  kAdvisories,
  kAdvisoriesSkipped,
  kAdvisoryRows,
  kLateAfterFatal,
  kNoVictim,
  kVersion,
  kRunning,
  kSpinCpu,
  kCounterCount,
};

inline uint32_t load_acquire(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release(uint8_t* address, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(address), value, __ATOMIC_RELEASE);
}

inline bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

inline int64_t record_offset(int64_t ring, uint32_t records, uint32_t seq) {
  return ring + static_cast<int64_t>((seq - 1u) % records) * kRecordBytes;
}

struct Request {
  uint32_t seq = 0;
  int64_t row = 0;
  uint32_t after = 0;
  bool armed = true;
  std::vector<int32_t> need;
  std::vector<int32_t> protect;
};

// Seqlock read: the writer stores the payload, fences, then the seq word last, so a
// record whose seq reads `expected` both before and after the payload is whole.
inline bool read_record(const uint8_t* record, uint32_t expected, Request* request) {
  if (load_acquire(record + kRecSeq) != expected) return false;
  uint16_t row, need, protect;
  std::memcpy(&row, record + kRecRow, 2);
  std::memcpy(&need, record + kRecNeedCount, 2);
  std::memcpy(&protect, record + kRecProtectCount, 2);
  std::memcpy(&request->after, record + kRecAfter, 4);
  uint32_t armed;
  std::memcpy(&armed, record + kRecArmed, 4);
  request->armed = armed != 0;
  request->seq = expected;
  request->row = row;
  const auto* need_ids = reinterpret_cast<const int32_t*>(record + kRecNeed);
  const auto* protect_ids = reinterpret_cast<const int32_t*>(record + kRecProtect);
  request->need.assign(need_ids, need_ids + std::min<int>(need, kMaxIds));
  request->protect.assign(protect_ids, protect_ids + std::min<int>(protect, kMaxIds));
  std::atomic_thread_fence(std::memory_order_acquire);
  return load_acquire(record + kRecSeq) == expected;
}

inline void set_status(uint8_t* record, uint16_t status) {
  __atomic_store_n(reinterpret_cast<uint16_t*>(record + kRecStatus), status, __ATOMIC_RELEASE);
}

inline bool listed(const std::vector<int32_t>& ids, int32_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

struct Tier {
  int64_t capacity = 0;
  std::vector<int32_t> slot_to_expert;
  std::vector<uint8_t> state;
  std::vector<uint64_t> stamp;
  std::vector<int32_t> expert_slot;  // assigned slot (LOADING or READY) or -1
  std::vector<uint8_t> hot;
  int64_t rows_demand = 0;
  int64_t rows_advisory = 0;
};

// The pinned-slot bookkeeping of every streamed layer (plan D12) and the service of one
// request at a time. pump_demand/pump_advice are called by one caller at a time: a test's
// pump(), or the Task 12 thread. The Python-facing methods take the same mutex.
class RamTier {
 public:
  RamTier(uint8_t* page, int32_t* slot_map, Tables tables, std::vector<int64_t> capacity, bool direct)
      : page_(page),
        map_(slot_map),
        layers_(tables.layers),
        experts_(tables.experts),
        reader_(std::move(tables), direct),
        tiers_(static_cast<size_t>(layers_)) {
    for (auto& counter : counters_)
      counter.store(0);
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      tier.capacity = capacity[row];
      tier.slot_to_expert.assign(tier.capacity, -1);
      tier.state.assign(tier.capacity, kFree);
      tier.stamp.assign(tier.capacity, 0);
      tier.expert_slot.assign(experts_, -1);
      tier.hot.assign(experts_, 0);
    }
  }

  bool open() {
    if (!reader_.open()) return false;
    next_demand_ = load_acquire(page_ + kDemandDone) + 1u;
    if (next_demand_ == 0) next_demand_ = 1;
    next_advice_ = load_acquire(page_ + kAdviseDone) + 1u;
    if (next_advice_ == 0) next_advice_ = 1;
    return true;
  }

  uint8_t* page() const {
    return page_;
  }
  int64_t busy_since() const {
    return busy_since_.load();
  }
  void set_counter(int index, int64_t value) {
    counters_[index].store(value);
  }
  void request_pause(bool paused) {
    pause_requested_.store(paused);
  }
  void request_stop(bool stopping) {
    stop_requested_.store(stopping);
  }
  void skip_advice_posted_so_far() {
    skip_advice_upto_.store(load_acquire(page_ + kAdviseHead));
  }
  bool threaded() const {
    return threaded_.load();
  }
  void set_threaded(bool threaded) {
    threaded_.store(threaded);
  }

  // Serve the next posted demand record, if any. True when it handled one.
  bool pump_demand() {
    const uint32_t head = load_acquire(page_ + kDemandHead);
    if (head == 0 || !reached(head, next_demand_)) return false;
    if (head - next_demand_ >= kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq.
      counters_[kOverruns].fetch_add(head - next_demand_ - (kDemandRecords - 2));
      next_demand_ = head - kDemandRecords + 2u;
    }
    uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);
    Request request;
    if (read_record(record, next_demand_, &request)) {
      handle_demand(request, record);
    } else {
      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop
    }
    _mm_sfence();
    store_release(page_ + kDemandDone, next_demand_);
    next_demand_ += 1u;
    return true;
  }

  // Serve (or skip) the next posted advisory record, if any. True when it handled one.
  bool pump_advice() {
    const uint32_t head = load_acquire(page_ + kAdviseHead);
    if (head == 0 || !reached(head, next_advice_)) return false;
    if (head - next_advice_ >= kAdviseRecords) {
      counters_[kAdvisoriesSkipped].fetch_add(head - next_advice_ - (kAdviseRecords - 2));
      next_advice_ = head - kAdviseRecords + 2u;
    }
    uint8_t* record = page_ + record_offset(kAdviseRing, kAdviseRecords, next_advice_);
    Request request;
    const uint32_t skip_upto = skip_advice_upto_.load();
    const bool stale = !read_record(record, next_advice_, &request) ||
                       (skip_upto != 0 && reached(skip_upto, next_advice_)) ||
                       reached(load_acquire(page_ + kDemandHead), request.after + 1u) ||
                       load_acquire(page_ + kFatal) != 0 || pause_requested_.load();
    if (stale) {
      counters_[kAdvisoriesSkipped].fetch_add(1);
    } else {
      in_advice_.store(true);
      counters_[kAdvisories].fetch_add(1);
      // An advisory gives up only between rows, not inside a blocking read: the watchdog's
      // stuck rule covers it like a demand, or a hung read would block stop()'s join forever.
      busy_since_.store(now_ns());
      int64_t rows = 0;
      serve(request, true, &rows);
      busy_since_.store(0);
      in_advice_.store(false);
    }
    store_release(page_ + kAdviseDone, next_advice_);
    next_advice_ += 1u;
    return true;
  }

  // ---- Python-facing bookkeeping; eager callers pause the thread first (Task 12) ----

  bool has(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    return tiers_[row].expert_slot[expert] >= 0;
  }

  void touch(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    const int32_t slot = tier.expert_slot[expert];
    if (slot >= 0) tier.stamp[slot] = ++tick_;
  }

  // A slot for a Python-side read; the map entry is published at once (the device is idle
  // and the thread paused when an eager path calls this). evicted: -1 none, -2 already held.
  int64_t assign(int64_t row, int64_t expert, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    if (tier.expert_slot[expert] >= 0) {
      *evicted = -2;
      return tier.expert_slot[expert];
    }
    const int64_t slot = take_slot_locked(row, protect, fallback, evicted);
    if (slot < 0) return -1;
    tier.slot_to_expert[slot] = static_cast<int32_t>(expert);
    tier.state[slot] = kReady;
    tier.stamp[slot] = ++tick_;
    tier.expert_slot[expert] = static_cast<int32_t>(slot);
    publish_map(row, expert, static_cast<int32_t>(slot));
    counters_[kVersion].fetch_add(1);
    return slot;
  }

  void release(int64_t row, int64_t slot) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (tiers_[row].state[slot] == kLoading) {
      // The service is filling it and will publish it; freeing it would hand it out twice.
      throw std::runtime_error("exl3 RAM miss: release of pinned slot " + std::to_string(slot) + " while it is loading");
    }
    release_locked(row, slot);
    counters_[kVersion].fetch_add(1);
  }

  void mapping(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const int32_t slot = tier.expert_slot[expert];
      out[expert] = slot >= 0 && tier.state[slot] == kReady ? slot : -1;
    }
  }

  void slot_to_expert(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot)
      out[slot] = tier.slot_to_expert[slot];
  }

  int64_t lru_order(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    std::vector<int64_t> slots;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kReady) slots.push_back(slot);
    }
    std::sort(slots.begin(), slots.end(), [&](int64_t a, int64_t b) { return tier.stamp[a] < tier.stamp[b]; });
    for (size_t i = 0; i < slots.size(); ++i)
      out[i] = tier.slot_to_expert[slots[i]];
    return static_cast<int64_t>(slots.size());
  }

  void set_hot(int64_t row, const int64_t* experts, int64_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    std::fill(tier.hot.begin(), tier.hot.end(), 0);
    for (int64_t i = 0; i < count; ++i) {
      if (experts[i] >= 0 && experts[i] < experts_) tier.hot[experts[i]] = 1;
    }
  }

  void layer_rows(int64_t* out, bool advisory) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (int64_t row = 0; row < layers_; ++row)
      out[row] = advisory ? tiers_[row].rows_advisory : tiers_[row].rows_demand;
  }

  // Test-only faults: sleep `delay_ns` before each advisory read and before each demand
  // read once `after_demands` demands have read rows; report reads as failed.
  void inject(int64_t delay_ns, bool fail_reads, int64_t after_demands) {
    delay_ns_.store(delay_ns);
    fail_reads_.store(fail_reads);
    delay_after_.store(after_demands);
  }

  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i)
      out[i] = counters_[i].load();
  }

 private:
  void publish_map(int64_t row, int64_t expert, int32_t slot) {
    __atomic_store_n(map_ + row * experts_ + expert, slot, __ATOMIC_RELEASE);
  }

  int64_t take_slot_locked(int64_t row, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
    Tier& tier = tiers_[row];
    *evicted = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kFree) return slot;
    }
    int64_t best = -1;
    int64_t spare = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] != kReady) continue;
      const int32_t expert = tier.slot_to_expert[slot];
      if (tier.hot[expert]) continue;
      if (listed(protect, expert)) {
        if (spare < 0 || tier.stamp[slot] < tier.stamp[spare]) spare = slot;
        continue;
      }
      if (best < 0 || tier.stamp[slot] < tier.stamp[best]) best = slot;
    }
    if (best < 0 && fallback) best = spare;
    if (best < 0) {
      counters_[kNoVictim].fetch_add(1);
      return -1;
    }
    const int32_t victim = tier.slot_to_expert[best];
    publish_map(row, victim, -1);  // unmapped before its bytes are overwritten (D11)
    tier.expert_slot[victim] = -1;
    tier.slot_to_expert[best] = -1;
    tier.state[best] = kFree;
    *evicted = victim;
    counters_[kEvictions].fetch_add(1);
    return best;
  }

  void release_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const int32_t expert = tier.slot_to_expert[slot];
    if (expert >= 0) {
      publish_map(row, expert, -1);
      tier.expert_slot[expert] = -1;
    }
    tier.slot_to_expert[slot] = -1;
    tier.state[slot] = kFree;
  }

  bool demand_pending() const {
    return !reached(next_demand_ - 1u, load_acquire(page_ + kDemandHead));
  }

  // An unarmed demand record: nobody waits on it, so the device may already be gathering
  // any mapped slot (the next token's rows too, once the thread lags). Only refresh the
  // recency of its assigned rows: no eviction, no read. False for an invalid record.
  bool touch_request(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return false;
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[request.row];
    for (const auto* ids : {&request.protect, &request.need}) {
      for (int32_t expert : *ids) {
        if (expert < 0 || expert >= experts_) return false;
        const int32_t slot = tier.expert_slot[expert];
        if (slot >= 0) tier.stamp[slot] = ++tick_;
      }
    }
    return true;
  }

  // An armed demand or an advisory: touch the request's assigned rows; read every protected
  // or needed expert that is not assigned (D12's recompute: the device is waiting on this
  // record, so no gather is in flight), evicting only unprotected, non-hot READY rows; publish.
  // An advisory protects only its own ids, reads one row at a time and gives up when a
  // demand is posted or a pause is requested (its rows so far are released).
  // *rows: the rows it read (0 when it failed).
  bool serve(const Request& request, bool advisory, int64_t* rows) {
    *rows = 0;
    std::vector<int32_t> wanted;
    for (const auto* ids : {&request.protect, &request.need}) {
      for (int32_t expert : *ids) {
        if (!listed(wanted, expert)) wanted.push_back(expert);  // one slot per expert (device bytes may repeat)
      }
    }
    std::vector<int32_t> missing;
    std::vector<int64_t> slots;
    bool ok = request.row >= 0 && request.row < layers_;
    if (ok) {
      std::lock_guard<std::mutex> guard(mutex_);
      Tier& tier = tiers_[request.row];
      for (int32_t expert : wanted) {
        if (expert < 0 || expert >= experts_) {
          ok = false;
          break;
        }
        const int32_t slot = tier.expert_slot[expert];
        if (slot >= 0) {
          tier.stamp[slot] = ++tick_;
        } else {
          missing.push_back(expert);
        }
      }
      for (size_t i = 0; ok && i < missing.size(); ++i) {
        int64_t evicted = -1;
        const int64_t slot = take_slot_locked(request.row, wanted, false, &evicted);
        if (slot < 0) {
          ok = false;
          break;
        }
        tier.slot_to_expert[slot] = missing[i];
        tier.state[slot] = kLoading;
        tier.expert_slot[missing[i]] = static_cast<int32_t>(slot);
        slots.push_back(slot);
      }
      if (!ok) {
        for (int64_t slot : slots)
          release_locked(request.row, slot);
        // Each slot taken may have evicted a row, and that eviction stays: the map moved.
        if (!slots.empty()) counters_[kVersion].fetch_add(1);
        slots.clear();
      }
    }
    if (ok && !missing.empty()) {
      const int64_t delay = delay_ns_.load();
      if (delay > 0 && (advisory || demands_read_ >= delay_after_.load())) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
      }
      if (fail_reads_.load()) {
        counters_[kReadErrors].fetch_add(1);
        ok = false;
      } else {
        const int result = reader_.read(request.row, missing, slots, advisory ? 1 : kBounceRows, [&] {
          return advisory && (demand_pending() || pause_requested_.load() || stop_requested_.load());
        });
        if (result == 0) counters_[kReadErrors].fetch_add(1);
        ok = result == 1;
      }
      if (!advisory) ++demands_read_;
      _mm_sfence();  // the split's memcpy stores land before the map publishes them (D11)
    }
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!slots.empty()) {
        Tier& tier = tiers_[request.row];
        for (size_t i = 0; i < slots.size(); ++i) {
          if (ok) {
            tier.state[slots[i]] = kReady;
            tier.stamp[slots[i]] = ++tick_;
            publish_map(request.row, missing[i], static_cast<int32_t>(slots[i]));
          } else {
            release_locked(request.row, slots[i]);
          }
        }
        if (ok) (advisory ? tier.rows_advisory : tier.rows_demand) += static_cast<int64_t>(slots.size());
        counters_[kVersion].fetch_add(1);
      }
    }
    if (ok) {
      *rows = static_cast<int64_t>(slots.size());
      counters_[kRowsRead].fetch_add(*rows);
      if (advisory) counters_[kAdvisoryRows].fetch_add(*rows);
    }
    return ok;
  }

  void handle_demand(const Request& request, uint8_t* record) {
    busy_since_.store(now_ns());
    store_release(page_ + kBusySeq, request.seq);
    if (load_acquire(page_ + kFatal) != 0) counters_[kLateAfterFatal].fetch_add(1);
    int64_t rows = 0;
    const bool ok = request.armed ? serve(request, false, &rows) : touch_request(request);
    // Classified by what was read: an empty need whose protect ids had to be read is D12's race.
    if (ok) counters_[rows == 0 ? kTouchOnly : kServedRequests].fetch_add(1);
    _mm_sfence();
    set_status(record, ok ? kServed : kFailed);
    store_release(page_ + kBusySeq, 0);
    busy_since_.store(0);
  }

  uint8_t* page_;
  int32_t* map_;
  int64_t layers_;
  int64_t experts_;
  RowReader reader_;
  std::vector<Tier> tiers_;
  std::mutex mutex_;
  uint64_t tick_ = 0;
  uint32_t next_demand_ = 1;
  uint32_t next_advice_ = 1;
  int64_t demands_read_ = 0;
  std::atomic<bool> in_advice_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> threaded_{false};
  std::atomic<uint32_t> skip_advice_upto_{0};
  std::atomic<int64_t> busy_since_{0};
  std::atomic<int64_t> delay_ns_{0};
  std::atomic<int64_t> delay_after_{0};
  std::atomic<bool> fail_reads_{false};
  std::atomic<int64_t> counters_[kCounterCount];
};

inline std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Shared ownership: every call holds its own reference, so a close() from another Python
// thread (or a finalizer) frees the service only after the calls in flight return.
inline std::unordered_map<int64_t, std::shared_ptr<RamTier>>& registry() {
  static std::unordered_map<int64_t, std::shared_ptr<RamTier>> tiers;
  return tiers;
}

inline std::shared_ptr<RamTier> find(int64_t handle) {
  std::lock_guard<std::mutex> guard(registry_mutex());
  const auto found = registry().find(handle);
  if (found == registry().end()) throw std::runtime_error("exl3 RAM miss: unknown handle");
  return found->second;
}

}  // namespace exl3_ram_miss

int64_t exl3_ram_miss_open(
    TensorView page,
    TensorView slot_map,
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    TensorView capacity,
    std::string paths,
    int64_t slot_bytes,
    int64_t direct) {
  using namespace exl3_ram_miss;
  const auto* capacity_data = static_cast<const int64_t*>(capacity.data_ptr());
  auto tier = std::make_shared<RamTier>(
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(slot_map.data_ptr()),
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, slot_bytes),
      std::vector<int64_t>(capacity_data, capacity_data + capacity.size(0)),
      direct != 0);
  if (!tier->open()) return -1;
  std::lock_guard<std::mutex> guard(registry_mutex());
  static int64_t next_handle = 1;
  const int64_t handle = next_handle++;
  registry().emplace(handle, std::move(tier));
  return handle;
}

// Defined after RamThread (the service thread block below).
void exl3_ram_miss_close(int64_t handle);

// 1 served a demand record, 2 an advisory record, 0 nothing posted. Refused while a thread pumps.
int64_t exl3_ram_miss_pump(int64_t handle) {
  const auto tier = exl3_ram_miss::find(handle);
  if (tier->threaded()) throw std::runtime_error("exl3 RAM miss: pump() while the service thread runs");
  if (tier->pump_demand()) return 1;
  return tier->pump_advice() ? 2 : 0;
}

int64_t exl3_ram_miss_contains(int64_t handle, int64_t row, int64_t expert) {
  return exl3_ram_miss::find(handle)->has(row, expert) ? 1 : 0;
}

void exl3_ram_miss_touch(int64_t handle, int64_t row, int64_t expert) {
  exl3_ram_miss::find(handle)->touch(row, expert);
}

void exl3_ram_miss_assign(
    int64_t handle, int64_t row, int64_t expert, TensorView protect, int64_t fallback, TensorView out) {
  auto* result = static_cast<int64_t*>(out.data_ptr());
  int64_t evicted = -1;
  result[0] = exl3_ram_miss::find(handle)->assign(row, expert, exl3_ram_miss::ids_of(protect), fallback != 0, &evicted);
  result[1] = evicted;
}

void exl3_ram_miss_release(int64_t handle, int64_t row, int64_t slot) {
  exl3_ram_miss::find(handle)->release(row, slot);
}

void exl3_ram_miss_mapping(int64_t handle, int64_t row, TensorView out) {
  exl3_ram_miss::find(handle)->mapping(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_slot_to_expert(int64_t handle, int64_t row, TensorView out) {
  exl3_ram_miss::find(handle)->slot_to_expert(row, static_cast<int64_t*>(out.data_ptr()));
}

int64_t exl3_ram_miss_lru_order(int64_t handle, int64_t row, TensorView out) {
  return exl3_ram_miss::find(handle)->lru_order(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_set_hot(int64_t handle, int64_t row, TensorView experts) {
  exl3_ram_miss::find(handle)->set_hot(row, static_cast<const int64_t*>(experts.data_ptr()), experts.size(0));
}

void exl3_ram_miss_inject(int64_t handle, int64_t delay_ns, int64_t fail_reads, int64_t after_demands) {
  exl3_ram_miss::find(handle)->inject(delay_ns, fail_reads != 0, after_demands);
}

void exl3_ram_miss_counters(int64_t handle, TensorView out) {
  exl3_ram_miss::find(handle)->counters(static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_layer_rows(int64_t handle, int64_t advisory, TensorView out) {
  exl3_ram_miss::find(handle)->layer_rows(static_cast<int64_t*>(out.data_ptr()), advisory != 0);
}

// ---- Host-side simulated device: the post and wait kernels' protocol, for CPU tests ----

int64_t exl3_ram_miss_sim_post(
    TensorView page, int64_t row, TensorView need, TensorView protect, int64_t advisory, int64_t after, int64_t armed) {
  using namespace exl3_ram_miss;
  auto* base = static_cast<uint8_t*>(page.data_ptr());
  const int64_t head_word = advisory ? kAdviseHead : kDemandHead;
  uint32_t seq = load_acquire(base + head_word) + 1u;
  if (seq == 0) seq = 1;
  uint8_t* record =
      base + record_offset(advisory ? kAdviseRing : kDemandRing, advisory ? kAdviseRecords : kDemandRecords, seq);
  const auto need_ids = ids_of(need);
  const auto protect_ids = ids_of(protect);
  const uint16_t row16 = static_cast<uint16_t>(row);
  const uint16_t need_count = static_cast<uint16_t>(std::min<size_t>(need_ids.size(), kMaxIds));
  const uint16_t protect_count = static_cast<uint16_t>(std::min<size_t>(protect_ids.size(), kMaxIds));
  const uint16_t pending = 0;
  const uint32_t after32 = static_cast<uint32_t>(after);
  const uint32_t armed32 = armed != 0 ? 1u : 0u;
  // Seqlock writer: invalidate seq, fence, payload, fence, seq last (a lapped record
  // still being rewritten can never carry a valid seq).
  store_release(record + kRecSeq, 0u);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  std::memset(record + 4, 0, kRecordBytes - 4);
  std::memcpy(record + kRecRow, &row16, 2);
  std::memcpy(record + kRecNeedCount, &need_count, 2);
  std::memcpy(record + kRecProtectCount, &protect_count, 2);
  std::memcpy(record + kRecStatus, &pending, 2);
  std::memcpy(record + kRecAfter, &after32, 4);
  std::memcpy(record + kRecArmed, &armed32, 4);
  if (need_count) std::memcpy(record + kRecNeed, need_ids.data(), 4 * need_count);  // data() may be null when empty
  if (protect_count) std::memcpy(record + kRecProtect, protect_ids.data(), 4 * protect_count);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  store_release(record + kRecSeq, seq);  // payload first, seq last (the seqlock order)
  store_release(base + head_word, seq);
  return seq;
}

// The wait kernel's decision rule: 1 served, 2 failed, 0 timed out (both raise fatal),
// 3 fatal already raised (the sticky fast path).
int64_t exl3_ram_miss_sim_wait(TensorView page, int64_t seq, int64_t timeout_ns) {
  using namespace exl3_ram_miss;
  auto* base = static_cast<uint8_t*>(page.data_ptr());
  const uint32_t want = static_cast<uint32_t>(seq);
  if (load_acquire(base + kFatal) != 0) return 3;
  const int64_t deadline = now_ns() + timeout_ns;
  auto raise_fatal = [&] {
    uint32_t zero = 0;
    __atomic_compare_exchange_n(
        reinterpret_cast<uint32_t*>(base + kFatal), &zero, want, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  };
  while (!reached(load_acquire(base + kDemandDone), want)) {
    if (now_ns() > deadline) {
      raise_fatal();
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
  const uint8_t* record = base + record_offset(kDemandRing, kDemandRecords, want);
  const uint16_t status = __atomic_load_n(reinterpret_cast<const uint16_t*>(record + kRecStatus), __ATOMIC_ACQUIRE);
  if (status == kServed) return 1;
  raise_fatal();
  return 2;
}

// Test only: a writer thread rewrites one record in a loop with the post kernel's seqlock
// order (seq = 0, fence, payload, fence, a new seq) while this thread reads it with
// read_record. out = {records accepted, accepted records whose payload is not their seq's}.
void exl3_ram_miss_seqlock_stress(int64_t duration_ns, TensorView out) {
  using namespace exl3_ram_miss;
  alignas(64) uint8_t record[kRecordBytes] = {};
  std::atomic<bool> done{false};
  const auto expected_ids = [](uint32_t round) { return static_cast<uint16_t>(round % kMaxIds + 1); };
  std::thread writer([&] {
    for (uint32_t round = 1; !done.load(std::memory_order_relaxed); ++round) {
      const uint16_t row = static_cast<uint16_t>(round), count = expected_ids(round);
      const int32_t id = static_cast<int32_t>(round);
      store_release(record + kRecSeq, 0u);
      std::atomic_thread_fence(std::memory_order_seq_cst);
      std::memset(record + 4, 0, kRecordBytes - 4);
      std::memcpy(record + kRecRow, &row, 2);
      std::memcpy(record + kRecNeedCount, &count, 2);
      std::memcpy(record + kRecProtectCount, &count, 2);
      std::memcpy(record + kRecAfter, &round, 4);
      for (int i = 0; i < count; ++i) {
        std::memcpy(record + kRecNeed + 4 * i, &id, 4);
        std::memcpy(record + kRecProtect + 4 * i, &id, 4);
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
      store_release(record + kRecSeq, round * kDemandRecords + 1u);  // seqs of one ring slot
    }
  });
  int64_t accepted = 0, torn = 0;
  const int64_t deadline = now_ns() + duration_ns;
  while (now_ns() < deadline) {
    const uint32_t seq = load_acquire(record + kRecSeq);
    Request request;
    if (seq == 0 || !read_record(record, seq, &request)) continue;
    ++accepted;
    const uint32_t round = (seq - 1u) / kDemandRecords;
    bool whole = request.after == round && request.row == static_cast<uint16_t>(round) &&
                 request.need.size() == expected_ids(round) && request.protect.size() == expected_ids(round);
    for (int32_t id : request.need)
      whole = whole && id == static_cast<int32_t>(round);
    for (int32_t id : request.protect)
      whole = whole && id == static_cast<int32_t>(round);
    if (!whole) ++torn;
  }
  done.store(true);
  writer.join();
  auto* result = static_cast<int64_t*>(out.data_ptr());
  result[0] = accepted;
  result[1] = torn;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_open, exl3_ram_miss_open);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_close, exl3_ram_miss_close);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pump, exl3_ram_miss_pump);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_contains, exl3_ram_miss_contains);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_touch, exl3_ram_miss_touch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_assign, exl3_ram_miss_assign);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_release, exl3_ram_miss_release);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_mapping, exl3_ram_miss_mapping);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_slot_to_expert, exl3_ram_miss_slot_to_expert);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_lru_order, exl3_ram_miss_lru_order);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_hot, exl3_ram_miss_set_hot);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject, exl3_ram_miss_inject);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_counters, exl3_ram_miss_counters);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_layer_rows, exl3_ram_miss_layer_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_sim_post, exl3_ram_miss_sim_post);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_sim_wait, exl3_ram_miss_sim_wait);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_seqlock_stress, exl3_ram_miss_seqlock_stress);

namespace exl3_ram_miss {

// Pumps one RamTier on its own thread (plan D19): demands first, then advisories; spins
// with _mm_pause() for spin_ns after the last request, else sleeps 50 us between polls.
// pause() is a handshake: it asks every advisory in flight to give up at its next row,
// skips advisories posted so far (resume() skips those posted during the pause), and returns once the loop has
// acknowledged the pause between two requests. While paused the loop takes no request, so an eager caller owns the
// slots until resume(). The watchdog (plan D15), on its own thread so a stuck read cannot silence it, aborts the
// process when the fatal word stays raised for fatal_wait without stop() (the process did not fail stop), or when one
// demand or advisory stays in service for fatal_wait (a hung read). It outlives the service thread's
// join in stop(), so a stop during a hung read, demand or advisory, still ends in its abort.
// pause()/resume() are not reentrant: their one owner is the slot table's depth counter
// (Task 14), which calls pause at depth 0->1 and resume at 1->0.
class RamThread {
 public:
  RamThread(std::shared_ptr<RamTier> tier, int cpu_core, int64_t fatal_wait_ns, int64_t spin_ns)
      : tier_(std::move(tier)),
        page_(tier_->page()),
        cpu_core_(cpu_core),
        fatal_wait_ns_(fatal_wait_ns),
        spin_ns_(spin_ns) {}

  ~RamThread() {
    stop();
  }

  // Throws when the thread cannot be pinned to cpu_core (it is then joined, never left floating).
  void start() {
    tier_->set_threaded(true);
    thread_ = std::thread([this] { run(); });
    while (pin_error_.load() == kPinPending)
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    if (const int error = pin_error_.load()) {
      stop_.store(true);
      thread_.join();
      tier_->set_threaded(false);
      throw std::runtime_error(
          "exl3 RAM miss: could not pin the service thread to core " + std::to_string(cpu_core_) + ": " +
          std::strerror(error));
    }
    watchdog_ = std::thread([this] { watch(); });
  }

  // The watchdog is stopped only after the service thread has joined: a join that blocks
  // on a hung read is then aborted by its stuck rule instead of hanging the process.
  void stop() {
    stop_.store(true);
    tier_->request_stop(true);  // an advisory in flight gives up at its next row
    if (thread_.joinable()) thread_.join();
    watch_stop_.store(true);
    if (watchdog_.joinable()) watchdog_.join();
    tier_->request_stop(false);
    tier_->set_threaded(false);
  }

  bool pause(int64_t timeout_ns) {
    tier_->request_pause(true);
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(true);
    const int64_t deadline = now_ns() + timeout_ns;
    while (!paused_.load()) {
      if (now_ns() > deadline) {
        resume();
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
  }

  // Advisories posted while paused predate the eager use: skip them too.
  void resume() {
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(false);
    tier_->request_pause(false);
  }

 private:
  void run() {
    int error = 0;
    if (cpu_core_ >= 0) {
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      CPU_SET(cpu_core_, &cpus);
      error = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    }
    tier_->set_counter(kSpinCpu, error != 0 ? -error : sched_getcpu());
    pin_error_.store(error);
    if (error != 0) return;
    tier_->set_counter(kRunning, 1);
    int64_t last_active = now_ns();
    uint32_t heartbeat = 0;
    uint32_t iterations = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
      // Not every iteration: the word shares the cache line the device polls.
      if ((++iterations & 1023u) == 1u) store_release(page_ + kHeartbeat, ++heartbeat);
      if (pause_requested_.load()) {
        paused_.store(true);
        while (pause_requested_.load() && !stop_.load())
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        paused_.store(false);
        continue;
      }
      if (tier_->pump_demand() || tier_->pump_advice()) {
        last_active = now_ns();
        iterations = 0;  // one heartbeat per request served
        continue;
      }
      if (now_ns() - last_active < spin_ns_) {
        _mm_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
    tier_->set_counter(kRunning, 0);
  }

  void watch() {
    int64_t fatal_since = 0;
    bool reported = false;
    while (!watch_stop_.load()) {
      const uint32_t fatal = load_acquire(page_ + kFatal);
      const int64_t now = now_ns();
      if (fatal != 0) {
        if (!reported) {
          reported = true;
          std::fprintf(stderr, "ERROR exl3 RAM miss: request %u timed out or failed; the process must stop\n", fatal);
          std::fflush(stderr);
        }
        if (fatal_since == 0) fatal_since = now;
      }
      const int64_t busy_since = tier_->busy_since();
      // Once stop() began, the process is failing stop: only a hung read can still abort.
      const bool fatal_held = !stop_.load() && fatal_since != 0 && now - fatal_since > fatal_wait_ns_;
      const bool stuck = busy_since != 0 && now - busy_since > fatal_wait_ns_;
      if (fatal_held || stuck) {
        std::fprintf(
            stderr,
            "ERROR exl3 RAM miss: %s for %.1f s (fatal %u, busy %u); aborting instead of hanging decode\n",
            stuck ? "a request stayed in service" : "the fatal word stayed raised without the process stopping",
            static_cast<double>(fatal_wait_ns_) / 1e9,
            fatal,
            load_acquire(page_ + kBusySeq));
        std::fflush(stderr);
        prctl(PR_SET_DUMPABLE, 0);
        std::abort();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  std::shared_ptr<RamTier> tier_;
  uint8_t* page_;
  int cpu_core_;
  int64_t fatal_wait_ns_;
  int64_t spin_ns_;
  std::thread thread_;
  std::thread watchdog_;
  static constexpr int kPinPending = -1;
  std::atomic<bool> stop_{false};
  std::atomic<bool> watch_stop_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> paused_{false};
  std::atomic<int> pin_error_{kPinPending};  // 0 pinned (or not asked), else the errno
};

// Guarded by registry_mutex(), like the tiers; shared for the same reason as the tiers.
inline std::unordered_map<int64_t, std::shared_ptr<RamThread>>& thread_registry() {
  static std::unordered_map<int64_t, std::shared_ptr<RamThread>> threads;
  return threads;
}

inline std::shared_ptr<RamThread> find_thread(int64_t handle) {
  std::lock_guard<std::mutex> guard(registry_mutex());
  const auto found = thread_registry().find(handle);
  if (found == thread_registry().end()) throw std::runtime_error("exl3 RAM miss: no service thread");
  return found->second;
}

}  // namespace exl3_ram_miss

void exl3_ram_miss_start_thread(int64_t handle, int64_t cpu_core, int64_t fatal_wait_ns, int64_t spin_ns) {
  using namespace exl3_ram_miss;
  if (cpu_core >= CPU_SETSIZE) throw std::runtime_error("exl3 RAM miss: cpu_core out of range");
  if (cpu_core >= 64 && cpu_core <= 71) {
    throw std::runtime_error("exl3 RAM miss: cores 64-71 are reserved (71 is production's doorbell core)");
  }
  if (cpu_core < 0) {
    cpu_set_t inherited;
    CPU_ZERO(&inherited);
    if (pthread_getaffinity_np(pthread_self(), sizeof(inherited), &inherited) == 0) {
      for (int core = 64; core <= 71; ++core) {
        if (CPU_ISSET(core, &inherited)) {
          std::fprintf(
              stderr,
              "WARNING exl3 RAM miss: the service thread inherits an affinity that includes reserved cores 64-71; "
              "run under taskset -c 0-63 or pass cpu_core\n");
          break;
        }
      }
    }
  }
  std::shared_ptr<RamTier> tier = find(handle);
  // Checked and registered under one lock, so a concurrent close() either sees the thread
  // (and joins it) or runs before it and leaves no handle to start it on.
  std::lock_guard<std::mutex> guard(registry_mutex());
  if (registry().count(handle) == 0) throw std::runtime_error("exl3 RAM miss: unknown handle");
  if (thread_registry().count(handle)) throw std::runtime_error("exl3 RAM miss: the service thread already runs");
  auto thread = std::make_shared<RamThread>(std::move(tier), static_cast<int>(cpu_core), fatal_wait_ns, spin_ns);
  thread->start();
  thread_registry()[handle] = std::move(thread);
}

void exl3_ram_miss_stop_thread(int64_t handle) {
  using namespace exl3_ram_miss;
  std::shared_ptr<RamThread> thread;
  {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto found = thread_registry().find(handle);
    if (found == thread_registry().end()) return;
    thread = std::move(found->second);
    thread_registry().erase(found);
  }
  thread->stop();
}

int64_t exl3_ram_miss_pause(int64_t handle, int64_t timeout_ns) {
  return exl3_ram_miss::find_thread(handle)->pause(timeout_ns) ? 1 : 0;
}

void exl3_ram_miss_resume(int64_t handle) {
  exl3_ram_miss::find_thread(handle)->resume();
}

// Takes the tier and its service thread out of the registries under one lock (so no
// start_thread can slip in between), then joins the thread: it holds a reference to the
// tier, which writes through raw addresses of Python-owned tensors that the caller
// releases after this returns.
void exl3_ram_miss_close(int64_t handle) {
  using namespace exl3_ram_miss;
  std::shared_ptr<RamThread> thread;
  std::shared_ptr<RamTier> tier;
  {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto running = thread_registry().find(handle);
    if (running != thread_registry().end()) {
      thread = std::move(running->second);
      thread_registry().erase(running);
    }
    const auto found = registry().find(handle);
    if (found != registry().end()) {
      tier = std::move(found->second);
      registry().erase(found);
    }
  }
  if (thread) thread->stop();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_start_thread, exl3_ram_miss_start_thread);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_stop_thread, exl3_ram_miss_stop_thread);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pause, exl3_ram_miss_pause);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_resume, exl3_ram_miss_resume);

}  // namespace sglang
