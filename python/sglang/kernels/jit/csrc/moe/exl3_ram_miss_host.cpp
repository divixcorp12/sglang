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
#include <sys/stat.h>
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
#include <utility>
#include <vector>

namespace sglang {
namespace exl3_ram_miss {

using tvm::ffi::TensorView;

// A bounce bank holds kBounceRows row slots and there are kBanks banks (kBounceSlots slots): a bank is
// the unit that is reused only once every I/O and packing reference to it has retired. Ring credit
// (kQueueDepth) is unrelated to both.
constexpr int kBounceRows = 8;
constexpr int kBanks = 2;
constexpr int kBounceSlots = kBanks * kBounceRows;
constexpr unsigned kQueueDepth = 16;
constexpr int64_t kPage = 4096;

inline int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

struct StageRecord;

// Clock reads taken for a trace record, so a test can show a disabled trace takes none. Written
// only when a record exists: with the trace off nothing here runs beyond the null check.
inline std::atomic<int64_t>& traced_clock_reads() {
  static std::atomic<int64_t> reads{0};
  return reads;
}

// The only way a trace stamp reads the clock; null (trace off) is a branch, never a clock read.
inline int64_t stamp(const StageRecord* trace) {
  if (trace == nullptr) return 0;
  traced_clock_reads().fetch_add(1, std::memory_order_relaxed);
  return now_ns();
}

constexpr int kMaxDrives = 4;
// Per-row and per-extent stamps live in fixed arrays: the record is copied out as one fixed-width row
// of int64 and pushed into a preallocated ring, so nothing on the completion path allocates. A
// request reads at most 2 * kMaxIds = 16 distinct experts (need and protect ids, 8 each) and a row
// issues at most two extents (one per mirror root in use), so 16 rows and 32 extents hold every
// request the wire format can carry. Anything past them is counted in rows_untraced /
// extents_untraced, never stamped and never allowed to grow the record.
constexpr int kTraceRows = 16;
constexpr int kTraceExtents = 32;

// One request's stage record, written only when the stage trace is on. Fixed size and int64
// words only, so it is copied out to Python as a row of a torch int64 tensor: keep
// STAGE_FIELDS in ops/moe/exl3_ram_miss.py in step. Every time is now_ns(), CLOCK_MONOTONIC on
// the host; a stage the request never reached stays 0. Nothing here is a GPU timestamp.
//
// Terminal status: how the request ended. kStatusNone (0) is never stored in a pushed record.
constexpr int64_t kStatusServed = 1;     // every missing row was read, packed and published
constexpr int64_t kStatusNoRead = 2;     // served with nothing to read: every needed row was resident
constexpr int64_t kStatusFailed = 3;     // an I/O error, a short file, an invalid request or no victim
constexpr int64_t kStatusCancelled = 4;  // an advisory gave up (demand posted, pause or stop)
constexpr int64_t kStatusTouch = 5;      // an unarmed demand: recency refreshed, no read possible

// Byte split (all in bytes, per request, summed over its io_uring batches):
//   useful_bytes     bytes copied into the slabs: sum of segment.bytes over every row packed. Each
//                    byte is counted once, so a retry never adds to it. 0 for rows not packed.
//   submitted_bytes  the length of every SQE prepared, first attempts and resubmissions. A submit
//                    that fails may leave some prepared SQEs the kernel never saw.
//   bytes            COMPLETED: the positive results of every completion reaped, including those of
//                    a batch that later failed. Never above submitted_bytes; below it by the
//                    aligned tail an extent asked for past end of file.
//   retried_bytes    the part of submitted_bytes that was a resubmission after -EINTR/-EAGAIN or a
//                    short read. submitted_bytes - retried_bytes is the first attempts' total.
//   cancelled_bytes  when a batch fails: the bytes its extents were expected to return (clamped at
//                    end of file) that never arrived, whether the extent was in flight, queued or
//                    errored. 0 for a batch that succeeds and for an advisory abandoned between
//                    batches (nothing is in flight then). On a failed batch, completed + cancelled
//                    is that batch's expected total.
// useful_bytes <= bytes <= submitted_bytes holds for a read that succeeds.
//
// Per-row packing: row_pack_start/end[k] bound the memcpy of row k of the request (k indexes the
// request's missing rows in read order, the same order as its slots). A row packs as soon as ITS
// extents have completed, whatever the others are doing, so rows pack in completion order, not
// necessarily in request order, and may pack while other rows are still being read: that overlap is
// what row_pack_start < another row's last extent_cqe shows. 0/0 is a row that never packed (the read
// failed or was cancelled before it). rows_asked is the number of rows the read was asked for, so a
// missing row is a k below it with 0/0. pack_start is the first row's start, pack_end the last row's
// end, pack_ns the sum of the rows' spans (the gaps between rows are waiting, not packing).
//
// Per-extent CQE: extent_id[k] = (row ordinal << 16) | part and extent_cqe[k] is the time the wait
// that reaped that extent's last completion returned; 0 is an extent that never completed. io_uring
// gives no per-completion time, so this is the reaping wait's return and not when the drive finished:
// CQEs reaped together share it. Slots fill in issue order (batch by batch), the first
// min(extents, kTraceExtents) are valid.
//
// Per-row admission and per-extent submit (schema 3), indexed like the two above:
//   row_admit[k]        the batch holding row k took a bounce bank and queued its extents; one clock
//                       read per batch, so the rows of a batch share it. 0: never admitted.
//   extent_submit[k]    when the extent's FIRST read was prepared as an SQE. The kernel sees it at the
//                       submit() of the same loop turn, before any wait: the handover to within one
//                       syscall, not a clock read around the submit itself.
//   extent_attempts[k]  resubmissions after the first (-EINTR/-EAGAIN or a short read).
// These are not in STAGE_ORDER on purpose: rows overlap, so no single order of stamps holds across
// rows. Compare a row's own stamps: row_admit <= its extents' submit <= their cqe <= its pack_start.
//
// lanes (schema 4): the planned lane count the device posted with this request (kRecLanes), so a layer's
// lanes per request can be read against its `row`. It counts RAM hits as well as the rows read: rows_asked
// is only what was missing. Not clamped to kMaxIds, so a plan wider than the lanes the service is asked
// for shows here.
//
// dropped_before: records the trace ring dropped, for being full, immediately before this one was
// pushed. A gap in `seq` cannot locate a loss on its own (a skipped advisory has no record either).
//
// The stamps submit, first_cqe and last_cqe cover the whole read: the first submit, the first completion
// returned and the last one returned; submit_to_first_cqe_ns and first_to_last_cqe_ns are those spans.
// Packing overlaps reading, so pack_start may precede last_cqe (that is the overlap, per row above);
// the order that holds is observed <= reserved <= submit <= first_cqe <= pack_start and
// last_cqe <= pack_end (the last completion belongs to a row that packs after it) and
// pack_end <= mapped <= done.
//
// Pipeline high-water marks: rows_reading_max is the most rows with I/O outstanding at once,
// pending_max the most SQEs prepared and not yet reaped (never above the ring credit), bank_stalls the
// number of times admitting the next batch had to wait for its bank's rows to pack.
struct StageRecord {
  int64_t seq = 0;
  int64_t kind = 0;  // kStageDemand, kStageAdvisory, kStageTouch
  int64_t row = 0;   // streamed row (index into the layer ids), not the layer id
  int64_t ok = 0;
  int64_t rows = 0;     // rows read
  int64_t batches = 0;  // io_uring batches the read used
  int64_t backlog = 0;  // records already posted behind this one when the service saw it
  int64_t prev_done = 0;  // `done` of the request served just before this one (0: the first)
  int64_t observed = 0;   // the service saw the record posted (first poll that found it)
  int64_t reserved = 0;   // slots reserved under the tier mutex
  int64_t submit = 0;     // just before the first io_uring submit
  int64_t first_cqe = 0;  // the call that returned the first completion, returned
  int64_t last_cqe = 0;   // the call that returned the last completion, returned
  int64_t pack_start = 0;  // the first row's packing started
  int64_t pack_end = 0;    // the last row's packing ended
  int64_t mapped = 0;  // slots marked READY and slot-map entries published
  int64_t done = 0;    // completion word stored: the device's wait can release
  int64_t submit_to_first_cqe_ns = 0;
  int64_t first_to_last_cqe_ns = 0;
  int64_t pack_ns = 0;  // the sum of the rows' packing spans
  int64_t bytes = 0;    // completed bytes, summed over drives (see the byte split)
  int64_t extents = 0;  // reads issued: one per row and root with a non-empty part
  int64_t drive_dev[kMaxDrives] = {};  // st_dev of the drive's filesystem; -1 folds several drives
  int64_t drive_bytes[kMaxDrives] = {};
  int64_t drive_extents[kMaxDrives] = {};
  int64_t status = 0;  // kStatus*
  int64_t rows_asked = 0;
  int64_t useful_bytes = 0;
  int64_t submitted_bytes = 0;
  int64_t retried_bytes = 0;
  int64_t cancelled_bytes = 0;
  int64_t rows_untraced = 0;     // rows past kTraceRows: packed but not stamped
  int64_t extents_untraced = 0;  // extents past kTraceExtents: read but not stamped
  int64_t rows_reading_max = 0;
  int64_t pending_max = 0;
  int64_t bank_stalls = 0;
  int64_t row_pack_start[kTraceRows] = {};
  int64_t row_pack_end[kTraceRows] = {};
  int64_t extent_id[kTraceExtents] = {};
  int64_t extent_cqe[kTraceExtents] = {};
  int64_t dropped_before = 0;
  int64_t row_admit[kTraceRows] = {};
  int64_t extent_submit[kTraceExtents] = {};
  int64_t extent_attempts[kTraceExtents] = {};
  int64_t lanes = 0;
};
static_assert(sizeof(StageRecord) % sizeof(int64_t) == 0, "StageRecord is int64 words only");
// A function, not a constexpr: test_exl3_ram_miss_device_args reads every constexpr as a page constant.
inline int64_t stage_words() {
  return sizeof(StageRecord) / sizeof(int64_t);
}
constexpr int64_t kStageDemand = 0;
constexpr int64_t kStageAdvisory = 1;
constexpr int64_t kStageTouch = 2;

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
  std::vector<std::string> source_paths;  // paths[f]'s source shard: a mirror is a copy of it
  std::vector<int64_t> file_sizes;  // the SOURCE size of every file, mirrors included
  std::vector<Read> extents;        // [layers][experts][parts]
  std::vector<int64_t> starts;      // [layers][experts]: where the row starts in its aligned superset
  std::vector<Segment> segments;
  int64_t need_end = 0;  // the row's last needed byte + 1, from its start: max(src + bytes) over segments
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
    const std::string& source_paths,
    int64_t slot_bytes) {
  Tables t;
  t.layers = extents.size(0);
  t.experts = extents.size(1);
  t.parts = extents.size(2);
  t.slot_bytes = slot_bytes;
  const auto split_lines = [](const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (true) {
      const size_t end = text.find('\n', start);
      lines.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
      if (end == std::string::npos) break;
      start = end + 1;
    }
    return lines;
  };
  t.paths = split_lines(paths);
  t.source_paths = split_lines(source_paths);
  if (t.source_paths.size() != t.paths.size() || static_cast<size_t>(file_sizes.size(0)) != t.paths.size()) {
    throw std::runtime_error("exl3 RAM miss: every file needs a size and the path of the source shard it copies");
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
  // The EOF guard (RowReader::admit_batch) decides a whole row from ONE of its parts: it reads that
  // part's `offset - dest` as the row's aligned base and its file size as the row's file size. Both are
  // true by construction of today's builder - exl3_ram_miss.py repeats one source size across all the
  // parts of a shard, and the mirror layout puts two files under a row only as two copies of the SAME
  // shard - but nothing in this file pinned either, and a builder that ever gave a row parts from
  // genuinely different files would arm the guard to clear a row against the wrong size. Checked here,
  // once per table, rather than per read.
  for (size_t base = 0; base + static_cast<size_t>(t.parts) <= t.extents.size();
       base += static_cast<size_t>(t.parts)) {
    const Read* head = nullptr;
    for (int64_t p = 0; p < t.parts && head == nullptr; ++p) {
      if (t.extents[base + static_cast<size_t>(p)].length > 0) head = &t.extents[base + static_cast<size_t>(p)];
    }
    if (head == nullptr) continue;  // a row nothing reads never reaches the guard
    for (int64_t p = 0; p < t.parts; ++p) {
      const Read& e = t.extents[base + static_cast<size_t>(p)];
      // Only the reading parts: a zero-length part is never submitted and its fields are unused, so
      // requiring anything of them would over-constrain the builder for no gain.
      if (e.length <= 0) continue;
      if (t.file_sizes[e.file] != t.file_sizes[head->file] ||
          e.offset - e.dest != head->offset - head->dest) {
        throw std::runtime_error(
            "exl3 RAM miss: a row's parts disagree on their aligned base or their file size, so the "
            "EOF guard cannot decide the row from part 0");
      }
    }
  }
  const auto* start_data = static_cast<const int64_t*>(starts.data_ptr());
  t.starts.assign(start_data, start_data + t.layers * t.experts);
  const auto* segment_data = static_cast<const int64_t*>(segments.data_ptr());
  t.segments.resize(static_cast<size_t>(segments.size(0)));
  for (size_t i = 0; i < t.segments.size(); ++i) {
    t.segments[i] = Segment{segment_data[4 * i], segment_data[4 * i + 1], segment_data[4 * i + 2], segment_data[4 * i + 3]};
    t.need_end = std::max(t.need_end, t.segments[i].src + t.segments[i].bytes);
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
  // Process each reaped batch of completions back to front. Nothing in the reader may
  // depend on delivery order, and the kernel gives no ordering guarantee across drives,
  // so this makes that requirement testable rather than assumed.
  bool reverse_cqes = false;
  // >0: cap outstanding reads to this many, below the ring's own depth. Production
  // constants keep a batch (kBounceRows rows) inside the ring, so credit never binds
  // there; this makes the refill path reachable from a test without resizing the ring.
  int64_t max_outstanding = 0;
  // Pipeline faults. pack_delay_ns sleeps inside every row's packing (a slow copy: the other bank's
  // completions pile up meanwhile). poison fills a bounce slot with a pattern when a row takes it and
  // with another when the row has packed, and scribbles every retired descriptor, so a row packed
  // before its reads landed, a bank reused early or a retired descriptor used again shows in the bytes
  // or crashes. stale_cqe_call: the k-th extent to retire (1-based) has its completion delivered AGAIN
  // once its descriptor is recycled for another extent. generation_start seeds the generation counter
  // (near 2^32 the counter wraps within a test). submit_short_call: that submit consumes nothing and
  // reports success. ordinal narrows the per-part faults to the row with that index in the request (-1: any).
  int64_t pack_delay_ns = 0;
  bool poison = false;
  int64_t stale_cqe_call = 0;
  int64_t generation_start = 0;
  int64_t submit_short_call = 0;
  int64_t ordinal = -1;
  // A slow drive, simulated: the completions of the row with this index in the request are withheld
  // from the reader (they were reaped from the CQ, so the kernel is done with them) until nothing else
  // is in flight or waiting to pack, then delivered. Cached buffered reads complete inside submit, so
  // without this no test can hold an extent outstanding while other rows pack (-1: none).
  int64_t hold_ordinal = -1;
};

// The fault tensor of the test entry points: 19 int64 words. The last two are not reader faults:
// abandon_after makes the entry point's abandon callback say stop once that many batches were admitted
// (0: never), and step (0: kBounceRows) is the faulted call's rows per batch. Keep the layout in step
// with _fault_tensor in ops/moe/exl3_ram_miss.py.
constexpr int64_t kFaultWords = 19;

inline ReadFault fault_from(const int64_t* f) {
  ReadFault fault;
  fault.submit_error = static_cast<int>(f[0]);
  fault.submit_call = f[1];
  fault.submit_first = f[2] != 0;
  fault.cqe_error = static_cast<int>(f[3]);
  fault.cqe_call = f[4];
  fault.part = f[5];
  fault.part_error = static_cast<int>(f[6]);
  fault.part_short = f[7];
  fault.reverse_cqes = f[8] != 0;
  fault.max_outstanding = f[9];
  fault.pack_delay_ns = f[10];
  fault.poison = f[11] != 0;
  fault.stale_cqe_call = f[12];
  fault.generation_start = f[13];
  fault.submit_short_call = f[14];
  fault.ordinal = f[15];
  fault.hold_ordinal = f[16];
  return fault;
}

inline void check_fault_words(TensorView fault) {
  if (fault.size(0) != kFaultWords) throw std::runtime_error("exl3 RAM miss: the fault tensor has the wrong length");
}

// Entry points' abandon callback: stop once `after` batches were admitted (0: never).
inline std::function<bool(size_t)> abandon_after(int64_t after) {
  return [after](size_t admitted) { return after > 0 && admitted >= static_cast<size_t>(after); };
}

// io_uring superset reads of whole expert rows into page-aligned bounce banks, then the
// per-name split into the pinned slabs (Exl3ShardRowSource.read's copies).
//
// Pipeline (plan Task 4). A read() call is split into batches of `step` rows; batch b fills bank
// b % kBanks. Every bounce slot is one row's aligned superset, and every extent has its own
// preallocated descriptor: descriptor (slot, part) is the SQE's user_data together with a generation,
// so a completion can only be attributed to the extent that is live in that descriptor NOW. The three
// resources are independent of each other:
//   * ring credit    `pending <= capacity` (queue_depth()): how many SQEs may be prepared and not yet
//                    reaped. It knows nothing about banks; a bank can hold more extents than the ring.
//   * banks          memory: kBanks * kBounceRows slots. A bank is handed to a new batch only once
//                    every row of its previous batch has PACKED (and so every extent has completed and
//                    retired): I/O and packing are the two references a bank holds, and both must be
//                    gone before the kernel may write into it again.
//   * reading rows   at most `max_reading_rows` rows with I/O outstanding (an advisory reads one).
// A row is packed as soon as ITS extents have completed, while other rows are still in flight, and
// every completed row is packed before the call returns. read() itself publishes nothing: the caller
// keeps the slots LOADING until read() returns 1, so no row is visible before the whole request is.
// Packing writes only into the caller's not-yet-published slots and only from a slot whose extents
// have all completed, so a failure leaves at most fully packed rows in unpublished slots, never a
// half-packed one, and the caller releases them.
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
    retired_ = 0;
    stale_armed_ = false;
    stale_waiting_ = false;
    if (fault.generation_start != 0) generation_ = static_cast<uint32_t>(fault.generation_start);
  }

  // Completions reaped over the reader's life (tests: a zero-length extent must add none).
  int64_t cqes() const { return cqes_; }
  // Completions that named no live descriptor, and generation counter wraps (tests).
  int64_t stale_cqes() const { return stale_cqes_; }
  int64_t generation_wraps() const { return generation_wraps_; }

  bool open() {
    for (const auto& path : t_.paths) {
      const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct_ ? O_DIRECT : 0));
      if (fd < 0) {
        std::fprintf(stderr, "ERROR exl3 RAM miss: open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
      }
      fds_.push_back(fd);
      struct stat st;
      const bool statted = fstat(fd, &st) == 0;
      const size_t file = fds_.size() - 1;
      // The table clamps every read at end of file against the SOURCE size (file_sizes), so a
      // copy of another size would otherwise be clamped, or over-read, into a short or stale row
      // that looks complete. Fail here, naming both files: with dozens of shards a bare
      // "size mismatch" does not say which copy is bad.
      if (statted && static_cast<int64_t>(st.st_size) != t_.file_sizes[file]) {
        throw std::runtime_error(
            "exl3 RAM miss: " + path + " has size " + std::to_string(st.st_size) + " bytes but its source " +
            t_.source_paths[file] + " has size " + std::to_string(t_.file_sizes[file]) +
            " bytes; the copy is incomplete or stale");
      }
      const int64_t dev = statted ? static_cast<int64_t>(st.st_dev) : -1;
      size_t drive = 0;
      while (drive < devs_.size() && devs_[drive] != dev) ++drive;
      if (drive == devs_.size()) devs_.push_back(dev);
      file_drive_.push_back(static_cast<uint8_t>(std::min<size_t>(drive, kMaxDrives - 1)));
    }
    for (size_t drive = 0; drive < devs_.size(); ++drive) {
      const size_t slot = std::min<size_t>(drive, kMaxDrives - 1);
      drive_dev_[slot] = drive < kMaxDrives ? devs_[drive] : -1;
    }
    if (posix_memalign(reinterpret_cast<void**>(&bounce_), kPage, static_cast<size_t>(kBounceSlots * t_.slot_bytes)) != 0) {
      bounce_ = nullptr;
      return false;
    }
    // Every buffer the pipeline uses is sized here, once: a descriptor per (bounce slot, part), a queue
    // that can hold each descriptor once (an extent waits in it at most once at a time), and completion
    // and resubmission lists bounded by the same count.
    const size_t extents = static_cast<size_t>(kBounceSlots) * static_cast<size_t>(t_.parts);
    // A completion carries its descriptor index in the low 32 bits of user_data and its generation in
    // the high 32 (see prepare() and process()). process() rejects an index past descs_.size(), but a
    // count that does not fit in 32 bits would truncate on the way OUT, so a completion would name a
    // different live descriptor and pass that check: bytes would be credited to the wrong extent.
    if (extents > 0xFFFFFFFFull) return false;
    descs_.assign(extents, ExtentDesc{});
    queue_.assign(extents, 0);
    completions_.reserve(extents + 1);
    held_.reserve(extents);
    again_.reserve(extents);
    if (io_uring_queue_init(queue_depth(), &ring_, 0) != 0) return false;
    ring_ready_ = true;
    return true;
  }

  // Read `experts` of streamed row `layer` into `slots`, `step` rows per io_uring batch (at most
  // kBounceRows, one bank). `abandon(batches admitted so far)` runs before each batch is admitted and
  // whenever the loop comes back to it; true stops admitting new batches. Rows already admitted are
  // reaped and packed, so nothing is in flight when read() returns.
  // Returns 1 when every row landed, 0 on an I/O error or short file, -1 when abandoned before every
  // batch was admitted. `packed`, when not null, is set to 1 for every row that was packed (with -1 those
  // rows are complete and the caller may keep them; with 0 the caller releases everything).
  // `max_reading_rows` caps the rows with I/O outstanding (an advisory reads one row at a time).
  // Every return leaves the ring empty: nothing in flight, nothing prepared (I1).
  //
  // `trace`, when not null, receives this read's stage stamps and per-drive bytes (StageRecord).
  // Null costs a branch per event and no clock read; the stamps only read the clock and add to
  // `trace`, so they cannot change what is submitted, reaped, drained or copied.
  //
  // This is the hot path and checks nothing: `layer`, `experts` and `slots` must be in
  // range and `experts.size() == slots.size()`. The service (Task 11) and
  // read_rows_once (Python) validate at their boundaries.
  int read(
      int64_t layer,
      const std::vector<int32_t>& experts,
      const std::vector<int64_t>& slots,
      size_t step,
      const std::function<bool(size_t)>& abandon,
      StageRecord* trace = nullptr,
      std::vector<uint8_t>* packed = nullptr,
      size_t max_reading_rows = SIZE_MAX) {
    if (!ring_ready_) return 0;
    step = std::max<size_t>(1, std::min<size_t>(step, kBounceRows));
    Call& c = c_;
    c = Call{};
    c.layer = layer;
    c.experts = &experts;
    c.slots = &slots;
    c.step = step;
    c.total = experts.size();
    c.batches = (c.total + step - 1) / step;
    c.max_reading = std::max<size_t>(1, max_reading_rows);
    // Credit is the ring's alone: banks and rows in flight do not enter it.
    c.capacity = fault_.max_outstanding > 0
                     ? std::min<unsigned>(queue_depth(), static_cast<unsigned>(fault_.max_outstanding))
                     : queue_depth();
    c.trace = trace;
    c.packed = packed;
    if (packed) packed->assign(c.total, 0);
    if (trace) trace->rows_asked = static_cast<int64_t>(c.total);
    reset_pipeline();
    held_.clear();
    while (true) {
      if (!c.failed) admit(abandon);
      if (!c.failed) refill();
      if (c.failed) break;
      const bool ready = has_ready();
      if (c.pending == 0 && !ready && held_.empty()) break;
      // Submit what was prepared before packing, so storage stays busy while the CPU copies; only
      // block for a completion when there is no complete row to pack.
      if (c.pending > 0 || (!ready && !held_.empty())) reap(ready);
      if (c.failed) break;
      pack_one();
    }
    // Nothing in flight and nothing to pack, yet a row was left unread or unpacked, or a batch was
    // neither admitted nor abandoned: the loop's own bookkeeping is wrong. Fail instead of returning a
    // row that was never read.
    if (!c.failed) {
      bool clean = c.reading_rows == 0 && c.queue_count == 0;
      for (int b = 0; b < kBanks; ++b) clean = clean && rows_busy_[b] == 0 && bank_live_[b] == 0;
      if (!clean || (!c.abandoned && c.next_batch < c.batches)) c.failed = true;
    }
    if (trace && c.first_seen != 0) {
      trace->submit = c.submitted;
      trace->first_cqe = c.first_seen;
      trace->last_cqe = c.last_seen;
      trace->submit_to_first_cqe_ns = c.first_seen - c.submitted;
      trace->first_to_last_cqe_ns = c.last_seen - c.first_seen;
    } else if (trace) {
      trace->submit = c.submitted;
    }
    if (c.failed) {
      account_unfinished();
      drain(c.pending);
      return 0;
    }
    return c.abandoned && c.next_batch < c.batches ? -1 : 1;
  }

 private:
  static constexpr int kMaxSoftErrors = 1000;
  static constexpr int kMaxRetries = 8;
  static constexpr uint8_t kPoisonFill = 0xA5;
  static constexpr int32_t kPoisonSlot = 0x7EADBEEF;

  enum class RowState : uint8_t { Free, Reading, Ready };

  // One extent's read, live from admission until its last completion retires it (generation != 0).
  struct ExtentDesc {
    const Read* read = nullptr;
    int64_t done = 0;
    int64_t expected = 0;
    uint32_t generation = 0;  // 0: retired, no completion may name it
    int32_t retries = 0;
    int32_t slot = -1;        // bounce slot: bank * kBounceRows + row within the bank
    int32_t trace_slot = -1;  // index into the record's extent arrays, -1 when not stamped
  };

  struct BounceRow {
    RowState state = RowState::Free;
    size_t ordinal = 0;      // the row's index in the request
    unsigned extents_left = 0;
    // Coverage, checked before the row is packed. `needed` is the last byte of the slot the segments
    // can read; `filled` is what the drives actually delivered into it. See pack_one().
    int64_t needed = 0;
    int64_t filled = 0;
  };

  struct Completion {
    uint64_t data;
    int res;
  };

  // One read() call's state. Everything the pipeline mutates lives here or in the members below, all
  // sized at open(); read() allocates nothing.
  struct Call {
    int64_t layer = 0;
    const std::vector<int32_t>* experts = nullptr;
    const std::vector<int64_t>* slots = nullptr;
    size_t step = 1;
    size_t total = 0;
    size_t batches = 0;
    size_t next_batch = 0;
    size_t max_reading = SIZE_MAX;
    size_t reading_rows = 0;  // rows with I/O outstanding
    unsigned capacity = 0;
    unsigned pending = 0;  // SQEs prepared and not yet reaped
    size_t queue_head = 0;
    size_t queue_count = 0;
    StageRecord* trace = nullptr;
    std::vector<uint8_t>* packed = nullptr;
    bool failed = false;
    bool abandoned = false;
    bool stalled = false;  // a batch is waiting for its bank to retire
    int soft_errors = 0;
    int64_t submitted = 0, first_seen = 0, last_seen = 0;
  };

  // How many reads may be outstanding at once. Credit-based preparation in refill() means
  // this bounds concurrency, not batch size: a batch larger than the ring waits for credit
  // rather than overrunning it. Scaled by parts so splitting a row across roots does not
  // halve the number of rows in flight.
  unsigned queue_depth() const { return kQueueDepth * static_cast<unsigned>(t_.parts); }

  uint32_t next_generation() {
    if (++generation_ == 0) {  // 0 means retired: skip it when the counter wraps
      ++generation_;
      ++generation_wraps_;
    }
    return generation_;
  }

  uint8_t* bounce_slot(size_t slot) const { return bounce_ + slot * static_cast<size_t>(t_.slot_bytes); }

  // A new read() starts with every descriptor retired and every bank free. This is also what makes a
  // failed call safe to follow: drain() has already retired the kernel's side of everything.
  void reset_pipeline() {
    for (auto& d : descs_) d = ExtentDesc{};
    for (auto& r : rows_) r = BounceRow{};
    for (int b = 0; b < kBanks; ++b) rows_busy_[b] = bank_live_[b] = 0;
  }

  bool has_ready() const {
    for (const auto& r : rows_) {
      if (r.state == RowState::Ready) return true;
    }
    return false;
  }

  void queue_push(uint32_t index) {
    Call& c = c_;
    // queue_ holds each descriptor at most once, so queue_count + pending <= descs_.size() == queue_.size():
    // a descriptor leaves the queue before it is prepared and only re-enters (through again_) after its
    // completion was reaped. If that ever broke, the modulo below would overwrite the queue's head and
    // silently drop an extent's read while its row still packed and published - the old native-bypass
    // bug's signature. Cheap enough to check on every push, and there is no safe way to continue.
    if (c.queue_count >= queue_.size()) {
      throw std::runtime_error("exl3 RAM miss: the extent queue overflowed its descriptor count");
    }
    queue_[(c.queue_head + c.queue_count) % queue_.size()] = index;
    ++c.queue_count;
  }

  // Admit batches while a bank is free. The abandon check comes first: once it says stop, no further
  // work is submitted, but what was already submitted is reaped by the loop.
  void admit(const std::function<bool(size_t)>& abandon) {
    Call& c = c_;
    while (!c.failed && !c.abandoned && c.next_batch < c.batches) {
      if (abandon(c.next_batch)) {
        c.abandoned = true;
        return;
      }
      const size_t first = c.next_batch * c.step;
      const size_t count = std::min(c.step, c.total - first);
      const size_t bank = c.next_batch % static_cast<size_t>(kBanks);
      if (rows_busy_[bank] != 0) {
        // The bank still holds rows that have not packed: the kernel must not write it again.
        if (!c.stalled && c.trace) ++c.trace->bank_stalls;
        c.stalled = true;
        return;
      }
      // The latch clears only once this turn really admits: returning below for the reading-rows cap
      // leaves the same busy bank to re-arm the edge next turn and count the SAME wait again. With the
      // advisory configuration (step 1, max_reading 1) that interleaving is the normal case, so one
      // wait spanning three turns would be reported as three stalls.
      if (c.reading_rows != 0 && c.reading_rows + count > c.max_reading) return;
      c.stalled = false;
      if (!admit_batch(bank, first, count)) {
        c.failed = true;
        return;
      }
      ++c.next_batch;
    }
  }

  bool admit_batch(size_t bank, size_t first, size_t count) {
    Call& c = c_;
    const size_t parts = static_cast<size_t>(t_.parts);
    if (bank_live_[bank] != 0) return false;  // an extent still names this bank: never reuse it
    // Validate the whole batch before touching any state, so a bad row leaves nothing to undo.
    for (size_t i = 0; i < count; ++i) {
      const size_t row_index = static_cast<size_t>(c.layer * t_.experts + (*c.experts)[first + i]);
      const size_t base = row_index * parts;
      // The bytes the expert needs are [start, start + need_end) of its aligned superset; they
      // must all lie inside the file. What an extent's page-aligned tail overruns past end of
      // file is padding no one needs, so the expectation below is shortened for it, but a row that
      // needs bytes the file does not have is corrupt and must fail, not publish the bounce's stale
      // bytes.
      // The head is the row's FIRST READING part, not part 0. A root whose split weight is 0 gives a
      // zero-length part 0 (SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=0:1 is a supported setting), and reading
      // the base and the file size out of an extent that is never submitted makes the guard depend on
      // fields nothing else uses: the eager reader already skips zero-length parts before computing
      // offsets (exl3_row_reader.py), so a builder that borrowed that idiom would leave them zeroed and
      // silently aim this check at the wrong file. Every part the head can be is one the reader submits.
      const Read* head = nullptr;
      for (size_t p = 0; p < parts && head == nullptr; ++p) {
        if (t_.extents[base + p].length > 0) head = &t_.extents[base + p];
      }
      // A row with no extent reads nothing, so packing it would publish the bounce's stale bytes.
      if (head == nullptr) return false;
      if (head->offset - head->dest + t_.starts[row_index] + t_.need_end > t_.file_sizes[head->file]) return false;
      // Every reading part must have at least one byte in its own file. This cannot happen with a
      // builder table - a part spans whole pages, a superset overruns end of file by less than one
      // page, so a reading part always starts strictly inside the file - which is exactly why it must
      // fail loudly here instead of being clamped to an expectation of zero below. An extent that
      // expects nothing retires having read nothing while its row still packs and publishes, which is
      // the corruption mode this reader exists to prevent.
      for (size_t p = 0; p < parts; ++p) {
        const Read& e = t_.extents[base + p];
        if (e.length > 0 && t_.file_sizes[e.file] - e.offset <= 0) return false;
      }
    }
    const int64_t admitted = stamp(c.trace);
    if (c.trace) ++c.trace->batches;
    for (size_t i = 0; i < count; ++i) {
      const size_t ordinal = first + i;
      if (c.trace && ordinal < static_cast<size_t>(kTraceRows)) c.trace->row_admit[ordinal] = admitted;
      const size_t slot = bank * kBounceRows + i;
      const size_t row_index = static_cast<size_t>(c.layer * t_.experts + (*c.experts)[ordinal]);
      const size_t base = row_index * parts;
      rows_[slot] = BounceRow{RowState::Reading, ordinal, 0};
      rows_[slot].needed = t_.starts[row_index] + t_.need_end;
      ++rows_busy_[bank];
      ++c.reading_rows;
      if (fault_.poison) std::memset(bounce_slot(slot), kPoisonFill, static_cast<size_t>(t_.slot_bytes));
      for (size_t p = 0; p < parts; ++p) {
        // A zero-length extent is a root that serves none of this row: no read, not queued.
        const Read* extent = &t_.extents[base + p];
        if (extent->length <= 0) continue;
        const uint32_t index = static_cast<uint32_t>(slot * parts + p);
        ExtentDesc& d = descs_[index];
        d = ExtentDesc{};
        d.read = extent;
        // Per extent, against the file that extent reads: a page-aligned tail overrunning end of file
        // is padding no one needs, so this expectation is shorter than the extent. At least 1 byte -
        // the validation loop above refused the batch if any reading extent started at or past EOF.
        d.expected = std::min(extent->length, t_.file_sizes[extent->file] - extent->offset);
        d.generation = next_generation();
        d.slot = static_cast<int32_t>(slot);
        if (stale_waiting_ && index == stale_index_) stale_armed_ = true;  // the fault's descriptor was just recycled
        ++rows_[slot].extents_left;
        ++bank_live_[bank];
        queue_push(index);
        if (c.trace) {
          const size_t drive = file_drive_[extent->file];
          c.trace->drive_dev[drive] = drive_dev_[drive];
          c.trace->drive_extents[drive] += 1;
          const int64_t trace_slot = c.trace->extents++;
          if (trace_slot < kTraceExtents) {
            c.trace->extent_id[trace_slot] = (static_cast<int64_t>(ordinal) << 16) | static_cast<int64_t>(p);
            d.trace_slot = static_cast<int32_t>(trace_slot);
          } else {
            ++c.trace->extents_untraced;
          }
        }
      }
    }
    if (c.trace) c.trace->rows_reading_max = std::max<int64_t>(c.trace->rows_reading_max, static_cast<int64_t>(c.reading_rows));
    return true;
  }

  // Prepare as many queued extents as credit and SQ room allow. `pending` counts SQEs prepared and not
  // yet reaped (in the SQ ring or in the kernel) and never exceeds `capacity`. Credits are counted by
  // nonempty extents, not by rows, because rows do not all issue the same number of reads: a root
  // serving none of a row issues nothing. A null SQE means the SQ filled before credit ran out: the
  // next submit sends what is prepared and refill runs again after the reap. Retries re-enter through
  // the queue, so they take credit like any other read.
  void refill() {
    Call& c = c_;
    int64_t prepared = 0;  // one clock read per refill turn, taken on the first SQE
    while (c.queue_count > 0 && c.pending < c.capacity) {
      io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
      if (sqe == nullptr) break;
      const uint32_t index = queue_[c.queue_head];
      c.queue_head = (c.queue_head + 1) % queue_.size();
      --c.queue_count;
      ExtentDesc& d = descs_[index];
      const int64_t remaining = d.read->length - d.done;
      io_uring_prep_read(
          sqe, fds_[d.read->file], bounce_slot(static_cast<size_t>(d.slot)) + d.read->dest + d.done,
          static_cast<unsigned>(remaining), static_cast<uint64_t>(d.read->offset + d.done));
      io_uring_sqe_set_data64(sqe, (static_cast<uint64_t>(d.generation) << 32) | index);
      ++c.pending;
      if (c.trace) {
        c.trace->submitted_bytes += remaining;
        if (d.done > 0 || d.retries > 0) c.trace->retried_bytes += remaining;
        c.trace->pending_max = std::max<int64_t>(c.trace->pending_max, static_cast<int64_t>(c.pending));
        if (d.trace_slot >= 0) {
          if (c.trace->extent_submit[d.trace_slot] == 0) {
            if (prepared == 0) prepared = stamp(c.trace);
            c.trace->extent_submit[d.trace_slot] = prepared;
          } else {
            ++c.trace->extent_attempts[d.trace_slot];
          }
        }
      }
    }
  }

  // Submit, wait for a completion only when `ready` is false, then drain the CQ before processing
  // it so the CQ frees early and a fault can reorder the completions.
  void reap(bool ready) {
    Call& c = c_;
    if (!ready && c.pending == 0 && !held_.empty()) {
      // Fault: every other row is done, so the withheld completions arrive now.
      completions_.assign(held_.begin(), held_.end());
      held_.clear();
      again_.clear();
      const int64_t released = stamp(c.trace);
      for (size_t k = 0; k < completions_.size(); ++k) process(completions_[k], released);
      if (c.trace) {
        if (c.first_seen == 0) c.first_seen = released;
        c.last_seen = released;
      }
      if (!c.failed) {
        for (uint32_t index : again_) queue_push(index);
      }
      return;
    }
    if (c.trace && c.submitted == 0) c.submitted = stamp(c.trace);
    const int rc = submit(ready ? 0u : 1u);
    if (rc < 0) {
      // -EINTR/-EAGAIN/-EBUSY: reap what has completed and submit again
      // (uring_file_reader.cpp). Anything else, or a soft error that never clears, fails.
      const bool soft = rc == -EINTR || rc == -EAGAIN || rc == -EBUSY;
      if (!soft || ++c.soft_errors > kMaxSoftErrors) {
        c.failed = true;
        return;
      }
    } else {
      c.soft_errors = 0;
    }
    const int64_t returned = stamp(c.trace);
    io_uring_cqe* cqe;
    unsigned head;
    unsigned seen = 0;
    completions_.clear();
    again_.clear();
    io_uring_for_each_cqe(&ring_, head, cqe) {
      ++seen;
      completions_.push_back(Completion{io_uring_cqe_get_data64(cqe), cqe->res});
    }
    io_uring_cq_advance(&ring_, seen);
    c.pending -= seen;
    if (fault_.reverse_cqes) std::reverse(completions_.begin(), completions_.end());
    if (fault_.hold_ordinal >= 0) {
      size_t kept = 0;
      for (size_t k = 0; k < completions_.size(); ++k) {
        const uint32_t index = static_cast<uint32_t>(completions_[k].data & 0xFFFFFFFFu);
        const bool live = index < descs_.size() && descs_[index].generation != 0 &&
                          descs_[index].generation == static_cast<uint32_t>(completions_[k].data >> 32);
        if (live && static_cast<int64_t>(rows_[descs_[index].slot].ordinal) == fault_.hold_ordinal) {
          held_.push_back(completions_[k]);
        } else {
          completions_[kept++] = completions_[k];
        }
      }
      completions_.resize(kept);
    }
    // Fault: a completion of an extent that retired earlier arrives after its descriptor was recycled.
    // It names a dead generation, so it must fail the read and touch nothing; without the generation
    // it would complete whichever extent now lives in that descriptor, publishing bytes never read.
    if (stale_armed_) {
      completions_.push_back(stale_);
      stale_armed_ = stale_waiting_ = false;
    }
    for (size_t k = 0; k < completions_.size(); ++k) process(completions_[k], returned);
    if (c.trace && seen > 0) {
      if (c.first_seen == 0) c.first_seen = returned;
      c.last_seen = returned;
    }
    if (c.failed) return;
    for (uint32_t index : again_) queue_push(index);
  }

  void process(const Completion& completion, int64_t returned) {
    Call& c = c_;
    const uint32_t index = static_cast<uint32_t>(completion.data & 0xFFFFFFFFu);
    const uint32_t generation = static_cast<uint32_t>(completion.data >> 32);
    if (index >= descs_.size() || generation == 0 || descs_[index].generation != generation) {
      ++stale_cqes_;
      c.failed = true;
      return;
    }
    ExtentDesc& d = descs_[index];
    const size_t part = index % static_cast<size_t>(t_.parts);
    int res = completion.res;
    ++cqes_;
    if (fault_.cqe_error != 0 && cqes_ == fault_.cqe_call) res = -fault_.cqe_error;
    if (fault_.part >= 0 && !part_fired_ && static_cast<int64_t>(part) == fault_.part &&
        (fault_.ordinal < 0 || static_cast<int64_t>(rows_[d.slot].ordinal) == fault_.ordinal)) {
      if (fault_.part_error != 0) {
        part_fired_ = true;
        res = -fault_.part_error;
      } else if (fault_.part_short > 0 && res > fault_.part_short) {
        part_fired_ = true;
        res = static_cast<int>(fault_.part_short);
      }
    }
    // The extent, not the row: two parts of one row complete independently.
    if (res == -EINTR || res == -EAGAIN) {
      if (++d.retries > kMaxRetries) {
        c.failed = true;
      } else {
        again_.push_back(index);  // resubmit the same range (M3)
      }
      return;
    }
    if (res < 0 || (res == 0 && d.done < d.expected)) {
      c.failed = true;
      return;
    }
    d.done += res;
    // A mid-file O_DIRECT read ends short only on a logical-block boundary, so
    // offset + done, bounce + dest + done and length - done stay block-aligned (an
    // extent's offset, dest and length are whole pages) and the resubmit is a legal
    // direct read of just this extent. At EOF, done == expected: no resubmit.
    if (d.done < d.expected) {
      again_.push_back(index);
      return;
    }
    retire(index, completion, returned);
  }

  // The extent's last completion: account it, retire the descriptor, and when it was its row's last
  // extent mark the row ready to pack. Nothing reads the descriptor afterwards.
  void retire(uint32_t index, const Completion& completion, int64_t returned) {
    Call& c = c_;
    ExtentDesc& d = descs_[index];
    if (c.trace) {
      const size_t drive = file_drive_[d.read->file];
      c.trace->drive_bytes[drive] += d.done;
      c.trace->bytes += d.done;
      if (d.trace_slot >= 0) c.trace->extent_cqe[d.trace_slot] = returned;
    }
    const size_t slot = static_cast<size_t>(d.slot);
    rows_[slot].filled += d.done;
    --bank_live_[slot / kBounceRows];
    if (fault_.stale_cqe_call > 0 && ++retired_ == fault_.stale_cqe_call) {
      stale_ = completion;
      stale_index_ = index;
      stale_waiting_ = true;
    }
    d = ExtentDesc{};
    if (fault_.poison) d.slot = kPoisonSlot;
    if (--rows_[slot].extents_left == 0) {
      rows_[slot].state = RowState::Ready;
      --c.reading_rows;
    }
  }

  // Pack ONE complete row, the earliest in request order among those ready. One row per loop turn
  // keeps packing bounded: the loop refills and reaps between rows.
  bool pack_one() {
    Call& c = c_;
    size_t best = kBounceSlots;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (rows_[s].state != RowState::Ready) continue;
      if (best == static_cast<size_t>(kBounceSlots) || rows_[s].ordinal < rows_[best].ordinal) best = s;
    }
    if (best == static_cast<size_t>(kBounceSlots)) return false;
    // Defence in depth for the one failure this reader must never have: packing bytes no drive
    // delivered. admit_batch's EOF guard already refuses a row the file cannot satisfy, but it decides
    // the row from part 0's file size alone, so it is only as good as the table's row consistency
    // (checked in tables_from). This compares what the drives actually returned for THIS row against
    // what its segments will read, costs one compare per row, and unlike the byte-split counters it is
    // not behind the trace flag. Extents fill the slot contiguously from dest 0 and only a tail extent
    // can stop short without being resubmitted (a short read retries; only the EOF clamp shortens an
    // expectation), so a total at least `needed` means the needed prefix is whole.
    if (rows_[best].filled < rows_[best].needed) {
      c.failed = true;
      return false;
    }
    const size_t ordinal = rows_[best].ordinal;
    const int64_t start = stamp(c.trace);
    if (fault_.pack_delay_ns > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(fault_.pack_delay_ns));
    // The row's parts landed contiguously, so its segments split from one base.
    const uint8_t* base =
        bounce_slot(best) + t_.starts[static_cast<size_t>(c.layer * t_.experts + (*c.experts)[ordinal])];
    const int64_t slot = (*c.slots)[ordinal];
    for (const Segment& segment : t_.segments) {
      std::memcpy(
          t_.slabs[c.layer][segment.name] + slot * t_.row_bytes[segment.name] + segment.dst, base + segment.src,
          static_cast<size_t>(segment.bytes));
    }
    if (c.trace) {
      const int64_t end = stamp(c.trace);
      if (ordinal < static_cast<size_t>(kTraceRows)) {
        c.trace->row_pack_start[ordinal] = start;
        c.trace->row_pack_end[ordinal] = end;
      } else {
        ++c.trace->rows_untraced;
      }
      for (const Segment& segment : t_.segments) c.trace->useful_bytes += segment.bytes;
      if (c.trace->pack_start == 0) c.trace->pack_start = start;
      c.trace->pack_end = std::max(c.trace->pack_end, end);
      c.trace->pack_ns += end - start;
    }
    if (c.packed) (*c.packed)[ordinal] = 1;
    // Packing is the last reference the bank held on this slot: only now may it be reused.
    if (fault_.poison) std::memset(bounce_slot(best), kPoisonFill ^ 0xFF, static_cast<size_t>(t_.slot_bytes));
    rows_[best] = BounceRow{};
    --rows_busy_[best / kBounceRows];
    return true;
  }

  // A failed call: what every extent still live was owed but never returned is cancelled. Extents that
  // retired were accounted when they did.
  void account_unfinished() {
    Call& c = c_;
    if (!c.trace) return;
    for (const ExtentDesc& d : descs_) {
      if (d.generation == 0) continue;
      const size_t drive = file_drive_[d.read->file];
      c.trace->drive_bytes[drive] += d.done;
      c.trace->bytes += d.done;
      c.trace->cancelled_bytes += std::max<int64_t>(0, d.expected - d.done);
    }
  }

  // Submit the prepared SQEs, waiting for `wait_nr` completions (0: do not block).
  int submit(unsigned wait_nr) {
    ++submits_;
    if (fault_.submit_error != 0 && submits_ == fault_.submit_call) {
      if (fault_.submit_first) io_uring_submit(&ring_);
      return -fault_.submit_error;
    }
    // Fault: the kernel consumed none of the prepared SQEs and reported success. They stay prepared and
    // are counted in `pending`, so the next submit must send them; nothing may wait on them meanwhile.
    if (fault_.submit_short_call != 0 && submits_ == fault_.submit_short_call) return 0;
    // A submit that consumes nothing while nothing is in flight would make the wait below block in
    // GETEVENTS for a completion no in-kernel SQE can produce (the state submit_short_call imitates).
    // Guarding it costs a second syscall on every batch of the decode path, and the service watchdog
    // already aborts a read that stays in service, so this is left to the watchdog deliberately.
    // If it ever does surface, the signature is busy_since_ non-zero with pending > 0 and an empty
    // completion queue; the guard would be to pass wait_nr = 0 whenever io_uring_sq_ready() > 0.
    return wait_nr != 0 ? io_uring_submit_and_wait(&ring_, wait_nr) : io_uring_submit(&ring_);
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
  std::vector<int64_t> devs_;       // st_dev of each distinct filesystem, in first-opened order
  std::vector<uint8_t> file_drive_;  // per file: its drive slot in a StageRecord
  int64_t drive_dev_[kMaxDrives] = {};
  ReadFault fault_{};
  int64_t submits_ = 0;
  int64_t cqes_ = 0;
  bool part_fired_ = false;
  // Pipeline state (see the class comment).
  Call c_;
  std::vector<ExtentDesc> descs_;
  std::vector<uint32_t> queue_;  // ring of descriptors waiting for credit; each appears at most once
  std::vector<Completion> completions_;
  std::vector<Completion> held_;  // fault: completions withheld from the reader (hold_ordinal)
  std::vector<uint32_t> again_;
  BounceRow rows_[kBounceSlots];
  size_t rows_busy_[kBanks] = {};   // rows not yet packed, per bank: the packing references
  size_t bank_live_[kBanks] = {};   // extents not yet retired, per bank: the I/O references
  uint32_t generation_ = 0;
  int64_t generation_wraps_ = 0;
  int64_t stale_cqes_ = 0;
  int64_t retired_ = 0;
  Completion stale_{0, 0};
  uint32_t stale_index_ = 0;
  bool stale_waiting_ = false;  // a retired completion is held until its descriptor is recycled
  bool stale_armed_ = false;    // ... and has been: deliver it with the next reap
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
    std::string source_paths,
    int64_t slot_bytes,
    int64_t direct,
    int64_t row,
    TensorView experts,
    TensorView slots,
    int64_t step) {
  using namespace exl3_ram_miss;
  RowReader reader(tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes), direct != 0);
  if (!reader.open()) return 0;
  return reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), [](size_t) { return false; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows, exl3_ram_miss_read_rows);

// Test only: exl3_ram_miss_read_rows with the reader's StageRecord copied to `record`
// (stage_words() int64), with `ok` and `status` set from the result. `fault` is the faulted call's
// tensor, laid out as exl3_ram_miss_read_rows_faulted's (kFaultWords words); an all-zero tensor injects nothing
// except that ordinal 0 selects row 0: the Python wrapper sends -1.
int64_t exl3_ram_miss_read_rows_traced(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    std::string source_paths,
    int64_t slot_bytes,
    int64_t direct,
    int64_t row,
    TensorView experts,
    TensorView slots,
    int64_t step,
    TensorView fault,
    TensorView record) {
  using namespace exl3_ram_miss;
  RowReader reader(tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes), direct != 0);
  if (!reader.open()) return 0;
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  reader.set_fault(fault_from(f));
  StageRecord stage;
  const int result = reader.read(
      row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), abandon_after(f[17]), &stage);
  stage.ok = result == 1 ? 1 : 0;
  stage.status = result == 1 ? kStatusServed : result == 0 ? kStatusFailed : kStatusCancelled;
  std::memcpy(record.data_ptr(), &stage, sizeof(stage));
  return result;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_traced, exl3_ram_miss_read_rows_traced);

// Test only: one reader reads `experts` into `slots` with `fault` injected
// (see ReadFault and fault_from), then reads `then_experts` into `then_slots` with no fault. Results go to
// `results[0..5]`: the two reads' results, the completions the reader had reaped after each, then its
// stale completions and generation wraps.
void exl3_ram_miss_read_rows_faulted(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    std::string source_paths,
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
  RowReader reader(tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes), direct != 0);
  if (!reader.open()) {
    out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = 0;
    return;
  }
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  reader.set_fault(fault_from(f));
  const size_t step = f[18] > 0 ? static_cast<size_t>(f[18]) : static_cast<size_t>(kBounceRows);
  out[0] = reader.read(row, ids_of(experts), slots_of(slots), step, abandon_after(f[17]));
  out[2] = reader.cqes();
  out[4] = reader.stale_cqes();
  out[5] = reader.generation_wraps();
  reader.set_fault(ReadFault{});
  out[1] = reader.read(row, ids_of(then_experts), slots_of(then_slots), kBounceRows, abandon_after(0));
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
// uint32: the layer's planned lane count as the device knew it when it posted (plan.count, RAM hits and
// misses together, not clamped to kMaxIds); an advisory carries the rows it asks for.
constexpr int64_t kRecLanes = 84;
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

// The device never posts sequence 0 (the post kernel and sim_post wrap 0xFFFFFFFF to 1), so a
// service that reaches 0 would spend an iteration on a record nobody posted and store a done word
// of 0.
inline uint32_t skip_zero(uint32_t seq) {
  return seq == 0 ? 1u : seq;
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
  uint32_t lanes = 0;
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
  std::memcpy(&request->lanes, record + kRecLanes, 4);
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

// Fixed-capacity single-producer single-consumer queue of stage records: the service thread
// pushes and drops (counted) when full, one Python caller at a time drains. Allocated once, when
// the trace is enabled; a push copies a record into a preallocated slot and allocates nothing.
class StageRing {
 public:
  explicit StageRing(size_t capacity) : slots_(capacity) {}

  void push(const StageRecord& record) {
    const uint64_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_.load(std::memory_order_acquire) >= slots_.size()) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      ++unreported_;
      return;
    }
    StageRecord& slot = slots_[head % slots_.size()];
    slot = record;
    slot.dropped_before = unreported_;
    unreported_ = 0;
    head_.store(head + 1, std::memory_order_release);
  }

  int64_t drain(StageRecord* out, int64_t max) {
    const uint64_t head = head_.load(std::memory_order_acquire);
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    int64_t count = 0;
    while (tail < head && count < max)
      out[count++] = slots_[tail++ % slots_.size()];
    tail_.store(tail, std::memory_order_release);
    return count;
  }

  int64_t dropped() const {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  std::vector<StageRecord> slots_;
  std::atomic<uint64_t> head_{0};
  std::atomic<uint64_t> tail_{0};
  std::atomic<int64_t> dropped_{0};
  int64_t unreported_ = 0;  // producer only: drops since the last record that got in
};

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
    begin_stage(kStageDemand, next_demand_, head - next_demand_);
    if (head - next_demand_ >= kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq.
      counters_[kOverruns].fetch_add(head - next_demand_ - (kDemandRecords - 2));
      next_demand_ = skip_zero(head - kDemandRecords + 2u);
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
    end_stage();
    next_demand_ = skip_zero(next_demand_ + 1u);
    return true;
  }

  // Serve (or skip) the next posted advisory record, if any. True when it handled one.
  bool pump_advice() {
    const uint32_t head = load_acquire(page_ + kAdviseHead);
    if (head == 0 || !reached(head, next_advice_)) return false;
    begin_stage(kStageAdvisory, next_advice_, head - next_advice_);
    if (head - next_advice_ >= kAdviseRecords) {
      counters_[kAdvisoriesSkipped].fetch_add(head - next_advice_ - (kAdviseRecords - 2));
      next_advice_ = skip_zero(head - kAdviseRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(kAdviseRing, kAdviseRecords, next_advice_);
    Request request;
    const uint32_t skip_upto = skip_advice_upto_.load();
    const bool stale = !read_record(record, next_advice_, &request) ||
                       (skip_upto != 0 && reached(skip_upto, next_advice_)) ||
                       reached(load_acquire(page_ + kDemandHead), request.after + 1u) ||
                       load_acquire(page_ + kFatal) != 0 || pause_requested_.load();
    if (stale) {
      cur_ = nullptr;  // a skipped advisory is no service: no stage record
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
    end_stage();
    next_advice_ = skip_zero(next_advice_ + 1u);
    return true;
  }

  // ---- Stage trace: one StageRecord per served request, drained by Python ----

  // Allocates the ring, then turns the trace on. Before the service thread starts, so the flag
  // never flips under a request being served.
  void enable_trace(size_t capacity) {
    if (threaded_.load()) throw std::runtime_error("exl3 RAM miss: enable the stage trace before the service thread starts");
    std::lock_guard<std::mutex> guard(trace_mutex_);
    ring_ = std::make_unique<StageRing>(capacity);
    trace_on_.store(true, std::memory_order_release);
  }

  // Up to `max` records into `out` (stage_words() int64 each); returns how many. The count of records
  // dropped for a full ring is `trace_dropped()`.
  int64_t drain_trace(StageRecord* out, int64_t max) {
    std::lock_guard<std::mutex> guard(trace_mutex_);
    return ring_ ? ring_->drain(out, max) : 0;
  }

  int64_t trace_dropped() {
    std::lock_guard<std::mutex> guard(trace_mutex_);
    return ring_ ? ring_->dropped() : 0;
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
  // read once `after_demands` demands have read rows; report reads as failed; make an advisory
  // give up once `abandon_after_batches` of its batches (rows) were admitted (0: never).
  void inject(int64_t delay_ns, bool fail_reads, int64_t after_demands, int64_t abandon_after_batches) {
    delay_ns_.store(delay_ns);
    fail_reads_.store(fail_reads);
    delay_after_.store(after_demands);
    abandon_after_.store(abandon_after_batches);
  }

  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i)
      out[i] = counters_[i].load();
  }

 private:
  // Called the moment a posted record is found. With the trace off this is one relaxed load and
  // no clock read; requests are served one at a time, so one member record serves them all.
  void begin_stage(int64_t kind, uint32_t seq, uint32_t backlog) {
    if (!trace_on_.load(std::memory_order_relaxed)) {
      cur_ = nullptr;
      return;
    }
    const int64_t observed = stamp(&stage_);  // before the reset below: the record is found, not built
    stage_ = StageRecord{};
    stage_.observed = observed;
    stage_.kind = kind;
    stage_.seq = seq;
    stage_.backlog = backlog;
    stage_.prev_done = last_done_;
    cur_ = &stage_;
  }

  void end_stage() {
    if (cur_ == nullptr) return;
    cur_->done = stamp(cur_);
    last_done_ = cur_->done;
    ring_->push(*cur_);
    cur_ = nullptr;
  }

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
  // An advisory protects only its own ids, has at most one row's I/O outstanding, and stops
  // submitting more when a demand is posted, a pause or a stop is requested. It reaps what it
  // already submitted, publishes the rows that completed and packed (each is a whole, valid row and
  // enters the tier as any READY row: evictable under the usual protection rules) and releases the rest.
  // A demand publishes nothing unless every row landed. *rows: the rows it read (0 when it failed).
  bool serve(const Request& request, bool advisory, int64_t* rows) {
    *rows = 0;
    if (cur_) cur_->lanes = request.lanes;
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
    if (cur_) cur_->reserved = stamp(cur_);
    int64_t status = kStatusNoRead;
    // Per slot: the row was packed whole (read() sets it). A member, not a local, so that the buffer
    // the pipeline writes into is allocated once rather than per served demand on the service thread
    // - read()'s assign() below only grows it, and it never shrinks.
    std::vector<uint8_t>& packed = packed_;
    // Cleared, not merely reused: the publish gate below reads packed[i] whenever the vector is long
    // enough, and read() only rewrites it when it actually runs. Carrying the PREVIOUS request's flags
    // into a request that never read would publish a row on the strength of an older row's packing.
    packed.clear();
    bool cancelled = false;
    if (ok && !missing.empty()) {
      const int64_t delay = delay_ns_.load();
      if (delay > 0 && (advisory || demands_read_ >= delay_after_.load())) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
      }
      if (fail_reads_.load()) {
        counters_[kReadErrors].fetch_add(1);
        ok = false;
        status = kStatusFailed;
      } else {
        const int64_t abandon_after = abandon_after_.load();
        const int result = reader_.read(
            request.row,
            missing,
            slots,
            advisory ? 1 : kBounceRows,
            [&](size_t admitted) {
              return advisory && (demand_pending() || pause_requested_.load() || stop_requested_.load() ||
                                  (abandon_after > 0 && admitted >= static_cast<size_t>(abandon_after)));
            },
            cur_,
            &packed,
            advisory ? 1 : SIZE_MAX);
        if (result == 0) counters_[kReadErrors].fetch_add(1);
        ok = result == 1;
        cancelled = result == -1;
        status = result == 1 ? kStatusServed : result == 0 ? kStatusFailed : kStatusCancelled;
      }
      if (!advisory) ++demands_read_;
      _mm_sfence();  // the split's memcpy stores land before the map publishes them (D11)
    }
    // A row is published only when it was packed whole: every row of a request that succeeded, and, for
    // a cancelled advisory, the rows that completed before it stopped. A failed request publishes none.
    int64_t published = 0;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!slots.empty()) {
        Tier& tier = tiers_[request.row];
        for (size_t i = 0; i < slots.size(); ++i) {
          if (ok || (cancelled && i < packed.size() && packed[i] != 0)) {
            tier.state[slots[i]] = kReady;
            tier.stamp[slots[i]] = ++tick_;
            publish_map(request.row, missing[i], static_cast<int32_t>(slots[i]));
            ++published;
          } else {
            release_locked(request.row, slots[i]);
          }
        }
        (advisory ? tier.rows_advisory : tier.rows_demand) += published;
        counters_[kVersion].fetch_add(1);
      }
    }
    if (cur_) {
      cur_->mapped = stamp(cur_);
      cur_->row = request.row;
      cur_->ok = ok ? 1 : 0;
      cur_->status = ok || status != kStatusNoRead ? status : kStatusFailed;
      cur_->rows = published;
    }
    if (published > 0) {
      counters_[kRowsRead].fetch_add(published);
      if (advisory) counters_[kAdvisoryRows].fetch_add(published);
    }
    if (ok) *rows = published;
    return ok;
  }

  void handle_demand(const Request& request, uint8_t* record) {
    busy_since_.store(now_ns());
    store_release(page_ + kBusySeq, request.seq);
    if (load_acquire(page_ + kFatal) != 0) counters_[kLateAfterFatal].fetch_add(1);
    int64_t rows = 0;
    const bool ok = request.armed ? serve(request, false, &rows) : touch_request(request);
    if (cur_ && !request.armed) {
      cur_->kind = kStageTouch;
      cur_->lanes = request.lanes;
      cur_->row = request.row;
      cur_->ok = ok ? 1 : 0;
      cur_->status = ok ? kStatusTouch : kStatusFailed;
    }
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
  std::vector<uint8_t> packed_;  // serve()'s per-row packed flags, sized by read(), reused every request
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
  std::atomic<int64_t> abandon_after_{0};
  std::atomic<bool> fail_reads_{false};
  std::atomic<int64_t> counters_[kCounterCount];
  // Stage trace. cur_ points at stage_ while a traced request is in service, else null.
  std::atomic<bool> trace_on_{false};
  std::mutex trace_mutex_;  // guards ring_ against a drain racing enable_trace
  std::unique_ptr<StageRing> ring_;
  StageRecord stage_{};
  StageRecord* cur_ = nullptr;
  int64_t last_done_ = 0;
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
    std::string source_paths,
    int64_t slot_bytes,
    int64_t direct) {
  using namespace exl3_ram_miss;
  const auto* capacity_data = static_cast<const int64_t*>(capacity.data_ptr());
  auto tier = std::make_shared<RamTier>(
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(slot_map.data_ptr()),
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes),
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

void exl3_ram_miss_inject(
    int64_t handle, int64_t delay_ns, int64_t fail_reads, int64_t after_demands, int64_t abandon_after_batches) {
  exl3_ram_miss::find(handle)->inject(delay_ns, fail_reads != 0, after_demands, abandon_after_batches);
}

void exl3_ram_miss_counters(int64_t handle, TensorView out) {
  exl3_ram_miss::find(handle)->counters(static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_layer_rows(int64_t handle, int64_t advisory, TensorView out) {
  exl3_ram_miss::find(handle)->layer_rows(static_cast<int64_t*>(out.data_ptr()), advisory != 0);
}

int64_t exl3_ram_miss_trace_words() {
  return exl3_ram_miss::stage_words();
}

void exl3_ram_miss_trace_enable(int64_t handle, int64_t capacity) {
  if (capacity <= 0) throw std::runtime_error("exl3 RAM miss: the stage trace needs a positive capacity");
  exl3_ram_miss::find(handle)->enable_trace(static_cast<size_t>(capacity));
}

// Fills up to out.size(0) records, stage_words() int64 each; returns the count.
int64_t exl3_ram_miss_trace_drain(int64_t handle, TensorView out) {
  return exl3_ram_miss::find(handle)->drain_trace(
      static_cast<exl3_ram_miss::StageRecord*>(out.data_ptr()), out.size(0));
}

int64_t exl3_ram_miss_trace_clock_reads() {
  return exl3_ram_miss::traced_clock_reads().load(std::memory_order_relaxed);
}

int64_t exl3_ram_miss_trace_dropped(int64_t handle) {
  return exl3_ram_miss::find(handle)->trace_dropped();
}

// ---- Host-side simulated device: the post and wait kernels' protocol, for CPU tests ----

int64_t exl3_ram_miss_sim_post(
    TensorView page,
    int64_t row,
    TensorView need,
    TensorView protect,
    int64_t advisory,
    int64_t after,
    int64_t armed,
    int64_t lanes) {
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
  const uint32_t lanes32 = static_cast<uint32_t>(std::max<int64_t>(0, lanes));
  std::memcpy(record + kRecLanes, &lanes32, 4);
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_words, exl3_ram_miss_trace_words);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_enable, exl3_ram_miss_trace_enable);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_drain, exl3_ram_miss_trace_drain);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_dropped, exl3_ram_miss_trace_dropped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_clock_reads, exl3_ram_miss_trace_clock_reads);
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
