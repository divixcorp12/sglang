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
#include <cassert>
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

#include "exl3_ram_miss_pack_pool.h"

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

// Piece streaming (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM, plan 2026-09-24-dsv41-piece-streaming §4.1-4.3). With
// it on, each nonzero part of a row is read as up to kSubReads page-aligned sub-reads, and the row's needed bytes are
// cut into kPieces pieces: piece j is sub-read j of the row (in file order) mapped into segment destination
// coordinates, its inner cuts rounded down to kPieceAlign. These are fixed, not knobs: the device's readiness word
// carries one bit per piece. With the flag off none of this is used and the reader issues one read per part.
constexpr int kSubReads = 4;
constexpr int kPieces = 8;
constexpr uint8_t kAllPieces = 0xFF;
constexpr int64_t kPieceAlign = 128;
static_assert(kPieces <= 8, "a row's piece and sub-read masks are one byte each");

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
// pack_workers, pack_split (schema 5): the packing mode the reader ran this request in, the reader's own
// pack_workers_ / pack_split_ (SGLANG_DSV41_RAM_MISS_PACK_WORKERS). pack_workers 0 is the inline reader:
// the owner thread packs each row itself, so a row's pack_start follows the extent's reap by however long the
// owner was busy, and pack_ns is a sum of spans that never overlap. pack_workers > 0 hands each row to a
// worker: pack_start is then when the worker had woken and taken a chunk, not when the packer was free, and
// the rows' spans overlap, so pack_ns can exceed pack_end - pack_start. A record does not say which of
// the two produced it without these, and a consumer that reads a worker record as an inline one reports
// wake-up latency as a busy packer and counts overlapping spans twice. pack_split is the chunks each row is
// cut into and means something only when pack_workers > 0. Every record carries the mode, including the ones
// no row was read for (no_read, touch): it is a property of the reader, not of the request.
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
//
// Piece streaming (schema 6). piece_stream is the reader's mode, carried like pack_workers. With it on, every
// extent is a sub-read and extent_id carries its index within its part in bits 8-15:
// (row ordinal << 16) | (sub << 8) | part; with it off sub is 0 and the id is the schema-5 one. Per row k (the first
// kTraceRows) and index j (sub-read ordinal in the row's file order, or piece):
//   sub_land_seq[k][j]  when sub-read j of row k retired (landed), as a sequence number (below);
//   piece_seq[k][j]     when piece j of row k was vetted: every sub-read in its dependency mask had landed and its
//                       bytes lie inside what they delivered;
//   piece_cqe[k][j]     the clock at that vetting: the reap that landed its last dependency (CQEs reaped together
//                       share it), or the row's admission for a piece with no bytes.
// The sequence numbers count landings and vettings together, 1-based, per read, in the order the owner saw them, so
// they order events that share one reap's stamp. 0: never happened. pieces_vetted counts every vetting of the read.
// All of these are 0 with the flag off. Vetting is not publishing: nothing here is visible to the device.
//
// Publishing (schema 7). Each piece is packed by its own job, and the owner publishes it once that job is done:
//   piece_publish[k][j]    when piece j of row k was published, on the same sequence as the landings and vettings;
//   pieces_published       pieces published (each once, however many readiness words name its row);
//   pieces_out_of_order    of those, the pieces published after a higher-numbered piece of their row;
//   piece_publish_refused  publish attempts a readiness word refused (another generation, or the bit already set).
// With piece streaming a row's row_pack_start/end span its pieces' jobs, so a row can start packing before its last
// sub-read lands.
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
  // With packing workers (SGLANG_DSV41_RAM_MISS_PACK_WORKERS) rows pack concurrently and a row's span starts at
  // its first chunk, after the worker woke: do not read overlap or "waited for the packer" out of these stamps.
  int64_t pack_start = 0;  // the earliest row's packing started
  int64_t pack_end = 0;    // the last row's packing ended
  int64_t mapped = 0;  // slots marked READY and slot-map entries published
  int64_t done = 0;    // completion word stored: the device's wait can release
  int64_t submit_to_first_cqe_ns = 0;
  int64_t first_to_last_cqe_ns = 0;
  int64_t pack_ns = 0;  // the sum of the rows' packing spans; with workers the spans overlap, so it can exceed pack_end - pack_start
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
  int64_t pack_workers = 0;
  int64_t pack_split = 0;
  int64_t piece_stream = 0;
  int64_t pieces_vetted = 0;
  int64_t sub_land_seq[kTraceRows][kPieces] = {};
  int64_t piece_cqe[kTraceRows][kPieces] = {};
  int64_t piece_seq[kTraceRows][kPieces] = {};
  int64_t piece_publish[kTraceRows][kPieces] = {};
  int64_t pieces_published = 0;
  int64_t pieces_out_of_order = 0;
  int64_t piece_publish_refused = 0;
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
  // With hold_ordinal set: withhold every row from that ordinal on, not only that row. They are released
  // together, so they become ready in one reap (a burst of rows the packer meets at once).
  bool hold_rest = false;
  // Piece streaming only: narrows the per-part faults to the sub-read with this index within its part (-1: any).
  // With hold_ordinal set it also narrows the hold to that sub-read (of part `part`, or of any part when part is
  // -1), so one sub-read of a row lands last while the row's others land. With the flag off every extent is sub 0.
  int64_t sub = -1;
  // Piece streaming only. publish_twice: the k-th piece the reader publishes (1-based, over its life) is published a
  // second time, as a re-dispatch would; the readiness word must refuse it. short_is_eof: the part_short completion
  // is also the end of its sub-read, as a file ending there would make it, so the sub-read retires with fewer bytes
  // than its pieces need.
  int64_t publish_twice = 0;
  bool short_is_eof = false;
  // Piece streaming, device tests. hold_until_probe_ms: pieces 1..7 of every row stay collected-but-unpublished until
  // the request's StreamProbe word reads tagged(1, generation) -- the stream kernel copied piece 0 -- or this many ms
  // pass (G2). last_publish_delay_ns: the read's last piece publish sleeps this long first, so it lands just before
  // kDemandDone (G11).
  int64_t hold_until_probe_ms = 0;
  int64_t last_publish_delay_ns = 0;
};

// A packing worker's chunk stamp: the same gated clock as every other stamp (a job is armed with it only
// for a traced read).
inline int64_t worker_stamp(const void* trace) {
  return stamp(static_cast<const StageRecord*>(trace));
}

// The fault tensor of the test entry points: 24 int64 words. Five of them are not reader faults:
// abandon_after makes the entry point's abandon callback say stop once that many batches were admitted
// (0: never), step (0: kBounceRows) is the faulted call's rows per batch, and pack_workers / pack_split
// configure the reader's packing pool before it opens (0 workers: pack inline on the owner; split 0:
// one chunk per worker); word 21 is hold_rest; word 22 (piece_stream, not a fault) turns the reader's piece
// streaming on before it opens; word 23 is sub, 24 publish_twice, 25 short_is_eof, 26 hold_until_probe_ms and 27
// last_publish_delay_ns. Keep the layout in step with _fault_tensor in ops/moe/exl3_ram_miss.py.
constexpr int64_t kFaultWords = 28;

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
  fault.hold_rest = f[21] != 0;
  fault.sub = f[23];
  fault.publish_twice = f[24];
  fault.short_is_eof = f[25] != 0;
  fault.hold_until_probe_ms = f[26];
  fault.last_publish_delay_ns = f[27];
  return fault;
}

inline void check_fault_words(TensorView fault) {
  if (fault.size(0) != kFaultWords) throw std::runtime_error("exl3 RAM miss: the fault tensor has the wrong length");
}

// Entry points' abandon callback: stop once `after` batches were admitted (0: never).
inline std::function<bool(size_t)> abandon_after(int64_t after) {
  return [after](size_t admitted) { return after > 0 && admitted >= static_cast<size_t>(after); };
}

// Piece streaming, sub-reads (plan §4.1): part `e` as its sub-reads, in file order, into `out` (kSubReads entries);
// returns how many. Each is len_k = round_up(ceil(length / kSubReads), kPage) bytes and the last takes what is left,
// so a part tiles exactly, a small part gives fewer than kSubReads and no sub-read is empty. A zero-length part gives
// none. The part's offset, dest and length are whole pages (the builder's), so every sub-read's are too.
inline int split_part(const Read& e, Read* out) {
  if (e.length <= 0) return 0;
  const int64_t len_k = ((e.length + kSubReads - 1) / kSubReads + kPage - 1) / kPage * kPage;
  int n = 0;
  for (int64_t at = 0; at < e.length && n < kSubReads; at += len_k) {
    out[n++] = Read{e.file, e.offset + at, std::min(len_k, e.length - at), e.dest + at};
  }
  return n;
}

// One segment's bytes [lo, hi), local to the segment: the same offsets in its source (src + lo) and its destination
// row (dst + lo), since a segment is one contiguous copy.
struct PieceRun {
  int64_t lo = 0;
  int64_t hi = 0;
};

// A row's sub-reads and pieces (plan §4.2), computed at admission from the row's start. Sub-read s (its ordinal in
// the row's file order) is part part[s]'s sub-read k[s]; piece j covers, in each segment i, the run
// runs[j * segments + i] of the caller's array, and depends on the sub-reads in deps[j].
struct RowGeometry {
  int subs = 0;
  int64_t start = 0;  // where the row's needed bytes begin in its bounce slot
  Read sub[kPieces] = {};
  int part[kPieces] = {};
  int k[kPieces] = {};
  uint8_t deps[kPieces] = {};
};

// Fill `g` and `runs` (kPieces * segments entries) for the row at `row_index` of the tables. Piece j's cuts are the
// row's sub-read boundaries: in every segment, piece j starts where sub-read j starts, mapped into the segment's
// destination coordinates and rounded down to kPieceAlign there (clamped to the segment), and ends where piece j + 1
// starts. Piece 0 starts at every segment's first byte and the last sub-read's piece ends at every segment's last,
// so the pieces partition the needed bytes; pieces past the row's sub-read count are empty. Segments are sorted by
// source offset (exl3_expert_format.py) and file order is dest order within a row (tables_from), so the cuts are
// monotone. deps[j] is exactly the set of sub-reads whose file bytes the piece's bytes touch. Returns false for a
// row that cannot be cut: more sub-reads than pieces, or a piece with bytes that no sub-read reads.
inline bool row_geometry(const Tables& t, size_t row_index, RowGeometry& g, PieceRun* runs) {
  const size_t parts = static_cast<size_t>(t.parts);
  const size_t base = row_index * parts;
  g = RowGeometry{};
  g.start = t.starts[row_index];
  for (size_t p = 0; p < parts; ++p) {
    Read split[kSubReads];
    const int n = split_part(t.extents[base + p], split);
    if (g.subs + n > kPieces) return false;
    for (int k = 0; k < n; ++k) {
      g.sub[g.subs] = split[k];
      g.part[g.subs] = static_cast<int>(p);
      g.k[g.subs] = k;
      ++g.subs;
    }
  }
  // Where piece j begins in segment `s`, as an offset local to the segment.
  const auto cut = [&](const Segment& s, int j) -> int64_t {
    if (j <= 0) return 0;
    if (j >= g.subs) return s.bytes;
    const int64_t at = g.sub[j].dest - g.start - s.src;  // the boundary, local to the segment
    if (at <= 0) return 0;
    if (at >= s.bytes) return s.bytes;
    return std::max<int64_t>(0, (s.dst + at) / kPieceAlign * kPieceAlign - s.dst);
  };
  const size_t segments = t.segments.size();
  for (int j = 0; j < kPieces; ++j) {
    bool bytes = false;
    for (size_t i = 0; i < segments; ++i) {
      const Segment& s = t.segments[i];
      const PieceRun run{cut(s, j), cut(s, j + 1)};
      runs[static_cast<size_t>(j) * segments + i] = run;
      if (run.lo >= run.hi) continue;
      bytes = true;
      const int64_t lo = g.start + s.src + run.lo, hi = g.start + s.src + run.hi;  // the run's bounce bytes
      for (int k = 0; k < g.subs; ++k) {
        if (lo < g.sub[k].dest + g.sub[k].length && g.sub[k].dest < hi) g.deps[j] |= static_cast<uint8_t>(1u << k);
      }
    }
    if (bytes && g.deps[j] == 0) return false;
  }
  return true;
}

// Piece streaming, publishing (plan §3.4). A readiness word (lease area P) is generation56 << 8 | bits8. The service
// initialises it to piece_word(generation) at reservation; the reader's owner then sets one bit per packed piece.
inline uint64_t piece_word(uint64_t generation) {
  return (generation & ((uint64_t{1} << 56) - 1)) << 8;
}

// Set `bit` in `word` only while the word still carries `generation` and does not have the bit: false (the word
// untouched) otherwise. A late publish from an older request fails the generation check instead of setting a bit
// under the new one, and a piece published twice fails the bit check. Release: the caller acquired the piece's
// packing (PackJob::done) before calling, so the device that acquires the bit sees the bytes.
inline bool publish_piece(uint64_t* word, uint64_t generation, uint8_t bit) {
  const uint64_t expected = generation & ((uint64_t{1} << 56) - 1);
  uint64_t old = __atomic_load_n(word, __ATOMIC_RELAXED);
  while (true) {
    if ((old >> 8) != expected || (old & bit) != 0) return false;
    if (__atomic_compare_exchange_n(word, &old, old | bit, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) return true;
  }
}

// Where the owner publishes a row's pieces: every readiness word naming the row, one per lane (at most kLeaseLanes,
// checked where the lease block is laid out). `rows` is indexed by the row's ordinal in the read.
constexpr int kPieceTargets = 8;
struct PieceTarget {
  uint64_t* words[kPieceTargets] = {};
  int count = 0;
};
struct PiecePublish {
  uint64_t generation = 0;
  const PieceTarget* rows = nullptr;
  const uint64_t* probe = nullptr;  // the request's StreamProbe word (read only by the hold_until_probe_ms fault)
};

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
// every completed row is packed before the call returns. read() itself publishes no row: the caller
// keeps the slots LOADING until read() returns 1, so no row is visible before the whole request is.
// With piece streaming each vetted piece is packed by its own job, and the owner publishes it into the
// caller's readiness words (PiecePublish) once that job is done; the slot map is still the caller's.
// Packing writes only into the caller's not-yet-published slots and only from a slot whose extents
// have all completed, so a failure leaves at most fully packed rows in unpublished slots, never a
// half-packed one, and the caller releases them. With piece streaming the unit is the piece: a piece is
// packed only once the sub-reads it depends on have landed and is published only once its job is done,
// so a failure leaves whole published pieces, never a torn one, in slots whose map the caller has not published. The
// caller quarantines each such slot a lane still leases (its lanes may be copying those pieces) and
// releases the rest.
class RowReader {
 public:
  RowReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0)
      : t_(std::move(tables)), direct_(direct) {
    set_pack(pack_workers, pack_split);
  }
  RowReader(const RowReader&) = delete;  // owns fds, the ring and the bounce
  RowReader& operator=(const RowReader&) = delete;

  ~RowReader() {
    pool_.reset();  // joins the workers before the bounce they read from is freed
    if (ring_ready_) io_uring_queue_exit(&ring_);
    for (int fd : fds_) ::close(fd);
    std::free(bounce_);
    // Undo the owner-pin scaffold's affinity change: the pin targets the calling thread, which a caller
    // (e.g. the benchmark) may reuse across many readers, so a later open() must see the original mask,
    // not the single core this reader pinned itself to.
    if (owner_pinned_) pthread_setaffinity_np(pthread_self(), sizeof(unpinned_affinity_), &unpinned_affinity_);
  }

  const Tables& tables() const { return t_; }

  // Pack on `workers` copy threads instead of the owner (0: on the owner, the default), each row in
  // `split` byte-range chunks (0: one per worker); with piece streaming each piece, about an eighth of a
  // row, is cut that way instead. Takes effect at open().
  void set_pack(int64_t workers, int64_t split) {
    pack_workers_ = static_cast<unsigned>(std::max<int64_t>(0, workers));
    pack_split_ = split > 0 ? static_cast<unsigned>(split) : pack_workers_;
  }
  unsigned pack_workers() const { return pack_workers_; }
  unsigned pack_split() const { return pack_split_; }
  bool pack_pool_active() const { return pool_ && pool_->active(); }
  std::vector<int> packing_cpus() const { return pool_ ? pool_->cpus() : std::vector<int>{}; }

  // Piece streaming (kSubReads sub-reads per part, per-piece vetting, packing and publishing); off by default. Before
  // open(), or on an idle reader after it (the tier sets it before its service thread starts), since it resizes the
  // descriptor arrays and the packing queue. Refused without packing workers (the inline path has no piece
  // publisher), with more mirror parts than the pieces can name, or when a slab row base is not kPieceAlign-aligned
  // (a piece's cuts are aligned in the row).
  void set_piece_stream(bool on) {
    if (on) {
      if (pack_workers_ == 0) throw std::runtime_error("exl3 RAM miss: piece streaming needs packing workers");
      if (t_.parts * kSubReads > kPieces) {
        throw std::runtime_error("exl3 RAM miss: piece streaming reads at most kPieces / kSubReads mirror parts");
      }
      for (size_t row = 0; row < t_.slabs.size(); ++row) {
        for (size_t name = 0; name < t_.slabs[row].size(); ++name) {
          if (reinterpret_cast<uintptr_t>(t_.slabs[row][name]) % kPieceAlign != 0 || t_.row_bytes[name] % kPieceAlign != 0) {
            throw std::runtime_error("exl3 RAM miss: piece streaming needs every slab row base 128 B aligned");
          }
        }
      }
    }
    piece_stream_ = on;
    subs_ = on ? kSubReads : 1;
    if (ring_ready_ && !size_extents()) throw std::runtime_error("exl3 RAM miss: too many descriptors for piece streaming");
    if (pool_) size_jobs();
  }
  bool piece_stream() const { return piece_stream_; }
  // Pieces a readiness word refused to publish, over the reader's life (each also failed its read).
  int64_t publish_refused() const { return publish_refused_; }
  // Test only (U10): the descriptor count and the ring credit this reader runs with.
  size_t descriptors() const { return descs_.size(); }
  unsigned credit() const { return queue_depth(); }
  // Test only (U10): every SQE prepared is appended here, while set (null: nothing recorded, one branch per SQE).
  struct SqeRecord {
    int64_t file, offset, length, bounce;  // bounce: byte offset of the destination from the bounce's start
  };
  void set_sqe_log(std::vector<SqeRecord>* log) { sqe_log_ = log; }

  // Test-only scaffold (PACK_WORKERS.md owner-pinning measurement): pin the owner thread to `core`
  // at open() and build the packing pool's mask as the inherited set minus that core, so the owner and
  // the workers never share a core. -1 (the default) leaves open() byte-for-byte what it is today: no
  // pin, and the pool's mask is exactly the creating thread's inherited affinity.
  void set_owner_core(int64_t core) { owner_core_ = core; }
  // Copies a worker still holds. read() leaves none: this is what a test checks after it returns.
  int64_t unfinished_jobs() const {
    int64_t open_jobs = 0;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (piece_stream_) {
        const uint8_t held = rows_[s].dispatched & static_cast<uint8_t>(~rows_[s].published);
        for (int j = 0; j < kPieces; ++j) {
          if ((held >> j & 1u) && !jobs_[s * kPieces + static_cast<size_t>(j)].done()) ++open_jobs;
        }
        continue;
      }
      if (rows_[s].state == RowState::Packing && !jobs_[s].done()) ++open_jobs;
    }
    return open_jobs;
  }

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
    if (!size_extents()) return false;
    if (io_uring_queue_init(queue_depth(), &ring_, 0) != 0) return false;
    ring_ready_ = true;
    cpu_set_t inherited;
    CPU_ZERO(&inherited);
    pthread_getaffinity_np(pthread_self(), sizeof(inherited), &inherited);
    if (owner_core_ >= 0) {
      unpinned_affinity_ = inherited;  // restored by the destructor
      CPU_CLR(static_cast<int>(owner_core_), &inherited);
      cpu_set_t owner_only;
      CPU_ZERO(&owner_only);
      CPU_SET(static_cast<int>(owner_core_), &owner_only);
      if (pthread_setaffinity_np(pthread_self(), sizeof(owner_only), &owner_only) != 0) {
        throw std::runtime_error("exl3 RAM miss: could not pin the owner thread to its core");
      }
      owner_pinned_ = true;
    }
    if (piece_stream_ && pack_workers_ == 0) throw std::runtime_error("exl3 RAM miss: piece streaming needs packing workers");
    if (pack_workers_ > 0) {
      // Every buffer the workers use is sized here too: a job and a run list per bounce slot.
      runs_.assign(static_cast<size_t>(kBounceSlots) * t_.segments.size(), CopyRun{});
      pool_ = std::make_unique<PackPool>(pack_workers_, inherited, static_cast<size_t>(kBounceSlots));
      if (piece_stream_) size_jobs();
    }
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
  // `progress`, when set, is invoked periodically from inside the drain loop -- at most once every
  // kProgressIntervalNs, never once per turn, since a turn can be as short as a single _mm_pause().
  // RowReader has no lease vocabulary and never will: this callback is how the caller (serve()) runs
  // its own periodic work (retire_leases()) while a read is in flight, exactly as `abandon` is how the
  // caller decides when to stop admitting. Null costs one comparison per turn and no clock read.
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
      size_t max_reading_rows = SIZE_MAX,
      const std::function<void()>& progress = nullptr,
      const PiecePublish* publish = nullptr) {
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
    c.publish = piece_stream_ ? publish : nullptr;
    if (c.publish != nullptr && c.publish->probe != nullptr && fault_.hold_until_probe_ms > 0) {
      c.hold_until = now_ns() + fault_.hold_until_probe_ms * 1000000;
    }
    if (packed) packed->assign(c.total, 0);
    if (trace) {
      trace->rows_asked = static_cast<int64_t>(c.total);
      trace->pack_workers = pack_workers_;
      trace->pack_split = pack_split_;
      trace->piece_stream = piece_stream_ ? 1 : 0;
    }
    reset_pipeline();
    held_.clear();
    // Whatever way this call ends, no packing worker may still be copying when it does: the caller
    // releases the slots on return and the next read reuses the bounce. Runs on exceptions too, and
    // parks the workers again, so none spins between reads.
    struct Quiesce {
      RowReader* reader;
      ~Quiesce() {
        reader->quiesce();
        if (reader->pool_) reader->pool_->set_active(false);
      }
    } quiesce_on_exit{this};
    if (pool_) pool_->set_active(true);  // workers wait for this read's jobs on their cores, not in the kernel
    // 0 forces the first turn to fire immediately, so a short read still gets one call before it
    // returns rather than waiting a full interval that may outlast the whole request.
    int64_t next_progress_ns = 0;
    while (true) {
      // Gated on elapsed time, not on a completion being reaped: reaped completions are this
      // reader's own I/O finishing, uncorrelated with the device acknowledging a lease (that arrives
      // through the lease block serve() owns), so a request whose reads finish early but is still
      // packing would otherwise stop calling progress() before the read returns. Time keeps firing
      // regardless of which sub-phase the loop is in. The interval is sized well under a typical
      // request's span (tens of ms, see the plan) so a lease is retired promptly, while staying far
      // above a single turn (as short as one _mm_pause()) so this never becomes a per-turn mutex take.
      if (progress) {
        const int64_t now = now_ns();
        if (now >= next_progress_ns) {
          progress();
          next_progress_ns = now + kProgressIntervalNs;
        }
      }
      collect_packed();  // before admit: a bank whose last copy just finished is free for the next batch
      if (!c.failed) admit(abandon);
      if (!c.failed) refill();
      if (c.failed) break;
      const bool ready = has_ready();
      if (c.pending == 0 && !ready && held_.empty() && c.packing == 0) break;
      // Submit what was prepared before packing, so storage stays busy while the CPU copies; only
      // block for a completion when there is no complete row to pack. With rows packing on workers the
      // owner cannot be woken from a blocking wait when one finishes, so it polls instead. The
      // withheld completions of the slow-drive fault arrive only once every other row has packed.
      if (c.pending > 0 || (!ready && c.packing == 0 && !held_.empty())) reap(ready || c.packing > 0);
      if (c.failed) break;
      if (!pack_one() && c.packing > 0) _mm_pause();
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
  // How often read()'s drain loop may call its optional `progress` callback. Arbitrary; chosen to sit
  // well under a request's typical span (tens of ms; see docs/superpowers/plans/2026-09-22-ram-miss-
  // progress-loop.md) so a lease is retired promptly, and far above one loop turn (a bare _mm_pause())
  // so the callback's own cost (a mutex and a walk over kDemandRecords, on the caller's side) cannot
  // dominate the loop.
  static constexpr int64_t kProgressIntervalNs = 200000;  // 200 us

  // Packing: handed to a packing worker, which owns the copy until the owner sees its job done.
  enum class RowState : uint8_t { Free, Reading, Ready, Packing };

  // One extent's read, live from admission until its last completion retires it (generation != 0).
  struct ExtentDesc {
    const Read* read = nullptr;
    int64_t done = 0;
    int64_t expected = 0;
    uint32_t generation = 0;  // 0: retired, no completion may name it
    int32_t retries = 0;
    int32_t slot = -1;        // bounce slot: bank * kBounceRows + row within the bank
    int32_t trace_slot = -1;  // index into the record's extent arrays, -1 when not stamped
    int32_t sub = 0;          // piece streaming: the sub-read's ordinal in its row's file order
  };

  struct BounceRow {
    RowState state = RowState::Free;
    size_t ordinal = 0;      // the row's index in the request
    unsigned extents_left = 0;
    // Coverage, checked before the row is packed. `needed` is the last byte of the slot the segments
    // can read; `filled` is what the drives actually delivered into it. See pack_one().
    int64_t needed = 0;
    int64_t filled = 0;
    // Piece streaming only (zero otherwise): the row's sub-reads by ordinal and its pieces. A piece is vetted once
    // every sub-read in its dependency mask has landed and its bytes lie inside what they delivered; with the flag
    // on this, not `filled`, is what take_ready_row checks.
    int64_t start = 0;  // where the needed bytes begin in the slot
    uint8_t subs = 0;
    uint8_t landed = 0;  // bit s: sub-read s retired
    uint8_t vetted = 0;  // bit j: piece j vetted
    uint8_t deps[kPieces] = {};
    int64_t sub_dest[kPieces] = {};
    int64_t sub_done[kPieces] = {};
    // Bit j: piece j handed to a packing job (or, with no bytes, published at once), and collected and published by
    // the owner. The row is finished once every piece is published and every sub-read retired.
    uint8_t dispatched = 0;
    uint8_t published = 0;
    int64_t pack_first = INT64_MAX;  // the earliest start and latest end of its pieces' jobs (traced reads only)
    int64_t pack_last = 0;
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
    size_t packing = 0;    // jobs handed to the packing workers and not yet collected by the owner (rows, or pieces)
    const PiecePublish* publish = nullptr;  // piece streaming: where the owner publishes (null: nowhere)
    int soft_errors = 0;
    int64_t submitted = 0, first_seen = 0, last_seen = 0;
    int64_t events = 0;  // piece streaming: sub-read landings and piece vettings so far (the trace's sequence)
    size_t published = 0;     // piece streaming: pieces published so far (the last_publish_delay_ns fault)
    int64_t hold_until = 0;   // piece streaming: when the hold_until_probe_ms fault gives up (0: no hold)
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

  // Every buffer the pipeline uses is sized here, once: a descriptor per (bounce slot, part, sub-read), a queue
  // that can hold each descriptor once (an extent waits in it at most once at a time), and completion
  // and resubmission lists bounded by the same count. With piece streaming off there is one sub-read per part,
  // so this is a descriptor per (bounce slot, part) and nothing piece-related is allocated.
  bool size_extents() {
    const size_t extents = static_cast<size_t>(kBounceSlots) * static_cast<size_t>(t_.parts) * subs_;
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
    sub_reads_.assign(piece_stream_ ? extents : 0, Read{});
    piece_runs_.assign(piece_stream_ ? static_cast<size_t>(kBounceSlots * kPieces) * t_.segments.size() : 0, PieceRun{});
    return true;
  }

  // A new read() starts with every descriptor retired and every bank free. This is also what makes a
  // failed call safe to follow: drain() has already retired the kernel's side of everything.
  void reset_pipeline() {
    for (auto& d : descs_) d = ExtentDesc{};
    for (auto& r : rows_) r = BounceRow{};
    for (int b = 0; b < kBanks; ++b) rows_busy_[b] = bank_live_[b] = 0;
  }

  // A row to pack; with piece streaming, a vetted piece not yet handed to a job, whatever its row's state.
  bool has_ready() const {
    for (const auto& r : rows_) {
      if (piece_stream_ ? (r.vetted & static_cast<uint8_t>(~r.dispatched)) != 0 : r.state == RowState::Ready) return true;
    }
    return false;
  }

  // Piece streaming packs a piece per job: a job and a run list per (bounce slot, piece), and a packing queue that
  // holds every one of them. With the flag off, a job and a run list per bounce slot, as open() sizes them.
  void size_jobs() {
    const size_t jobs = static_cast<size_t>(kBounceSlots) * (piece_stream_ ? kPieces : 1);
    runs_.assign(jobs * t_.segments.size(), CopyRun{});
    pool_->set_capacity(jobs);
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
    // Nor may a row still be packing (a worker holds its copy) or ready in one of its slots: rows_busy_ says so,
    // and this refuses if the two ever disagree instead of overwriting a row a worker is reading.
    for (size_t i = 0; i < count; ++i) {
      if (rows_[bank * kBounceRows + i].state != RowState::Free) return false;
    }
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
      // The slot is Free (checked above), so its piece runs may be written before the batch is known good.
      if (piece_stream_ && !plan_pieces(bank * kBounceRows + i, row_index, geometry_[i])) return false;
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
      if (piece_stream_) {
        queue_sub_reads(slot, ordinal, geometry_[i], admitted);
        continue;
      }
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

  // Piece streaming: the row's sub-reads and pieces, into `g` and slot `slot`'s piece runs. False refuses the batch:
  // the row cannot be cut, or a sub-read would start at or past end of file (the per-part check above, per sub-read;
  // a builder table never gives one, so it fails loudly rather than expecting nothing).
  bool plan_pieces(size_t slot, size_t row_index, RowGeometry& g) {
    const size_t segments = t_.segments.size();
    if (!row_geometry(t_, row_index, g, &piece_runs_[slot * kPieces * segments])) return false;
    for (int s = 0; s < g.subs; ++s) {
      if (t_.file_sizes[g.sub[s].file] - g.sub[s].offset <= 0) return false;
    }
    return true;
  }

  // Piece streaming: queue the row's sub-reads, one descriptor each, (slot, part, k) -> (slot * parts + part) *
  // kSubReads + k. Credit is untouched: refill() takes it per SQE, so a sub-read costs one like a part did. Pieces
  // with no bytes are vetted here, at admission.
  void queue_sub_reads(size_t slot, size_t ordinal, const RowGeometry& g, int64_t admitted) {
    Call& c = c_;
    const size_t parts = static_cast<size_t>(t_.parts);
    const size_t bank = slot / kBounceRows;
    BounceRow& r = rows_[slot];
    r.start = g.start;
    r.subs = static_cast<uint8_t>(g.subs);
    std::copy(g.deps, g.deps + kPieces, r.deps);
    for (int s = 0; s < g.subs; ++s) {
      const uint32_t index = static_cast<uint32_t>((slot * parts + static_cast<size_t>(g.part[s])) * kSubReads + g.k[s]);
      sub_reads_[index] = g.sub[s];
      const Read* extent = &sub_reads_[index];
      ExtentDesc& d = descs_[index];
      d = ExtentDesc{};
      d.read = extent;
      // Clamped at end of file per sub-read, as a part is; at least 1 byte (plan_pieces).
      d.expected = std::min(extent->length, t_.file_sizes[extent->file] - extent->offset);
      d.generation = next_generation();
      d.slot = static_cast<int32_t>(slot);
      d.sub = s;
      if (stale_waiting_ && index == stale_index_) stale_armed_ = true;  // the fault's descriptor was just recycled
      r.sub_dest[s] = extent->dest;
      ++r.extents_left;
      ++bank_live_[bank];
      queue_push(index);
      if (c.trace) {
        const size_t drive = file_drive_[extent->file];
        c.trace->drive_dev[drive] = drive_dev_[drive];
        c.trace->drive_extents[drive] += 1;
        const int64_t trace_slot = c.trace->extents++;
        if (trace_slot < kTraceExtents) {
          c.trace->extent_id[trace_slot] =
              (static_cast<int64_t>(ordinal) << 16) | (static_cast<int64_t>(g.k[s]) << 8) | static_cast<int64_t>(g.part[s]);
          d.trace_slot = static_cast<int32_t>(trace_slot);
        } else {
          ++c.trace->extents_untraced;
        }
      }
    }
    vet_pieces(slot, admitted);
  }

  // Piece streaming: sub-read `sub` of the row in `slot` retired with `done` bytes. Vet every piece it completes.
  void land_sub_read(size_t slot, int32_t sub, int64_t done, int64_t returned) {
    Call& c = c_;
    BounceRow& r = rows_[slot];
    r.landed |= static_cast<uint8_t>(1u << sub);
    r.sub_done[sub] = done;
    const int64_t seq = ++c.events;
    if (c.trace && r.ordinal < static_cast<size_t>(kTraceRows)) c.trace->sub_land_seq[r.ordinal][sub] = seq;
    vet_pieces(slot, returned);
  }

  // Vet each piece of `slot` whose dependencies have all landed and that is not vetted yet. A piece whose bytes the
  // landed sub-reads do not cover fails the call: its dependencies are final, so it can never be packed.
  void vet_pieces(size_t slot, int64_t when) {
    Call& c = c_;
    BounceRow& r = rows_[slot];
    for (int j = 0; j < kPieces; ++j) {
      const uint8_t bit = static_cast<uint8_t>(1u << j);
      if ((r.vetted & bit) != 0 || (r.deps[j] & ~r.landed) != 0) continue;
      if (!piece_delivered(slot, j)) {
        c.failed = true;
        return;
      }
      r.vetted |= bit;
      const int64_t seq = ++c.events;
      if (c.trace) {
        ++c.trace->pieces_vetted;
        if (r.ordinal < static_cast<size_t>(kTraceRows)) {
          c.trace->piece_cqe[r.ordinal][j] = when;
          c.trace->piece_seq[r.ordinal][j] = seq;
        }
      }
    }
  }

  // Every byte of piece j lies inside a landed sub-read's [dest, dest + done). Sub-reads are in dest order, so one
  // pass per run suffices. This replaces the row's `filled >= needed`: a sub-read short at end of file is covered
  // only up to what it returned.
  bool piece_delivered(size_t slot, int j) const {
    const BounceRow& r = rows_[slot];
    const size_t segments = t_.segments.size();
    const PieceRun* runs = &piece_runs_[(slot * kPieces + static_cast<size_t>(j)) * segments];
    for (size_t i = 0; i < segments; ++i) {
      if (runs[i].lo >= runs[i].hi) continue;
      int64_t at = r.start + t_.segments[i].src + runs[i].lo;
      const int64_t end = r.start + t_.segments[i].src + runs[i].hi;
      for (int s = 0; s < r.subs; ++s) {
        if (((r.landed >> s) & 1u) && r.sub_dest[s] <= at && at < r.sub_dest[s] + r.sub_done[s]) {
          at = r.sub_dest[s] + r.sub_done[s];
        }
      }
      if (at < end) return false;
    }
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
      if (sqe_log_) {
        sqe_log_->push_back(SqeRecord{
            d.read->file, d.read->offset + d.done, remaining,
            static_cast<int64_t>(d.slot) * t_.slot_bytes + d.read->dest + d.done});
      }
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
        if (live && (fault_.hold_rest ? static_cast<int64_t>(rows_[descs_[index].slot].ordinal) >= fault_.hold_ordinal
                                       : static_cast<int64_t>(rows_[descs_[index].slot].ordinal) == fault_.hold_ordinal) &&
            (fault_.sub < 0 || fault_matches_sub(index))) {
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
    // A fault's part is the descriptor's part whatever the sub-read count (subs_ is 1 with the flag off).
    const size_t part = (index / subs_) % static_cast<size_t>(t_.parts);
    int res = completion.res;
    bool eof = false;  // fault (short_is_eof): this completion ends the sub-read
    ++cqes_;
    if (fault_.cqe_error != 0 && cqes_ == fault_.cqe_call) res = -fault_.cqe_error;
    if (fault_.part >= 0 && !part_fired_ && static_cast<int64_t>(part) == fault_.part &&
        (fault_.sub < 0 || static_cast<int64_t>(index % subs_) == fault_.sub) &&
        (fault_.ordinal < 0 || static_cast<int64_t>(rows_[d.slot].ordinal) == fault_.ordinal)) {
      if (fault_.part_error != 0) {
        part_fired_ = true;
        res = -fault_.part_error;
      } else if (fault_.part_short > 0 && res > fault_.part_short) {
        part_fired_ = true;
        res = static_cast<int>(fault_.part_short);
        eof = fault_.short_is_eof;
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
    if (eof) d.expected = d.done;
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

  // Fault (hold_ordinal with sub): the completion is of sub-read fault_.sub of part fault_.part (any part at -1).
  bool fault_matches_sub(uint32_t index) const {
    const int64_t part = static_cast<int64_t>((index / subs_) % static_cast<size_t>(t_.parts));
    return static_cast<int64_t>(index % subs_) == fault_.sub && (fault_.part < 0 || part == fault_.part);
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
    if (piece_stream_) land_sub_read(slot, d.sub, d.done, returned);
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

  // The earliest row in request order among those ready, vetted for packing; kBounceSlots when there is
  // none or the vetting failed the call. Runs on the owner, before any copy, however the copy is done.
  size_t take_ready_row() {
    Call& c = c_;
    size_t best = kBounceSlots;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (rows_[s].state != RowState::Ready) continue;
      if (best == static_cast<size_t>(kBounceSlots) || rows_[s].ordinal < rows_[best].ordinal) best = s;
    }
    if (best == static_cast<size_t>(kBounceSlots)) return best;
    // Defence in depth for the one failure this reader must never have: packing bytes no drive
    // delivered. admit_batch's EOF guard already refuses a row the file cannot satisfy, but it decides
    // the row from part 0's file size alone, so it is only as good as the table's row consistency
    // (checked in tables_from). This compares what the drives actually returned for THIS row against
    // what its segments will read, costs one compare per row, and unlike the byte-split counters it is
    // not behind the trace flag. Extents fill the slot contiguously from dest 0 and only a tail extent
    // can stop short without being resubmitted (a short read retries; only the EOF clamp shortens an
    // expectation), so a total at least `needed` means the needed prefix is whole.
    // Piece streaming never takes a whole row: vet_pieces makes the same check per piece (dispatch_ready_pieces).
    if (rows_[best].filled < rows_[best].needed) {
      c.failed = true;
      return static_cast<size_t>(kBounceSlots);
    }
    return best;
  }

  // Pack ONE complete row inline, the earliest in request order among those ready. One row per loop turn
  // keeps packing bounded: the loop refills and reaps between rows. With a packing pool, every ready row
  // is handed to the workers instead, and the loop finishes each one when its copy is done.
  bool pack_one() {
    if (pool_) return piece_stream_ ? dispatch_ready_pieces() : dispatch_ready_rows();
    const size_t best = take_ready_row();
    if (best == static_cast<size_t>(kBounceSlots)) return false;
    Call& c = c_;
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
    finish_row(best, start, stamp(c.trace));
    return true;
  }

  // Hand every ready row to the packing workers. The row was vetted by take_ready_row on this thread; from
  // here until its job is done the workers own the copy and this thread must not touch its bounce slot.
  bool dispatch_ready_rows() {
    Call& c = c_;
    bool any = false;
    while (true) {
      const size_t best = take_ready_row();
      if (best == static_cast<size_t>(kBounceSlots)) return any;
      const size_t ordinal = rows_[best].ordinal;
      const uint8_t* base =
          bounce_slot(best) + t_.starts[static_cast<size_t>(c.layer * t_.experts + (*c.experts)[ordinal])];
      const int64_t slot = (*c.slots)[ordinal];
      // A slot's job is free only once its previous copy is done; arming it earlier would hand a worker a
      // half-armed job. Like queue_push's overflow, this cannot happen unless the accounting above is wrong.
      if (!jobs_[best].done()) throw std::runtime_error("exl3 RAM miss: a packing job was re-armed while a worker still holds it");
      CopyRun* runs = &runs_[best * t_.segments.size()];
      for (size_t i = 0; i < t_.segments.size(); ++i) {
        const Segment& segment = t_.segments[i];
        runs[i] = CopyRun{
            t_.slabs[c.layer][segment.name] + slot * t_.row_bytes[segment.name] + segment.dst, base + segment.src,
            segment.bytes};
      }
      jobs_[best].arm(
          runs, t_.segments.size(), pack_split_, fault_.pack_delay_ns, c.trace ? &worker_stamp : nullptr, c.trace);
      pool_->post(&jobs_[best]);  // throws before queueing: a row is Packing only once its job is posted
      rows_[best].state = RowState::Packing;
      ++c.packing;
      any = true;
    }
  }

  // Piece streaming: hand every vetted piece to its own packing job, whatever its row's state; the sub-reads it
  // depends on have landed (vet_pieces), and no read in flight writes its bounce bytes, since any that did would be
  // one of them. A piece with no bytes has nothing to store, so it is published at once. From here until its job is
  // done the workers own the copy.
  bool dispatch_ready_pieces() {
    Call& c = c_;
    const size_t segments = t_.segments.size();
    bool any = false;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots) && !c.failed; ++s) {
      BounceRow& r = rows_[s];
      const uint8_t todo = r.vetted & static_cast<uint8_t>(~r.dispatched);
      if (todo == 0) continue;
      const uint8_t* base = bounce_slot(s) + r.start;
      const int64_t slot = (*c.slots)[r.ordinal];
      for (int j = 0; j < kPieces && !c.failed; ++j) {
        const uint8_t bit = static_cast<uint8_t>(1u << j);
        if ((todo & bit) == 0) continue;
        const size_t job_index = s * kPieces + static_cast<size_t>(j);
        const PieceRun* piece = &piece_runs_[job_index * segments];
        CopyRun* runs = &runs_[job_index * segments];
        int64_t bytes = 0;
        for (size_t i = 0; i < segments; ++i) {
          const Segment& segment = t_.segments[i];
          runs[i] = CopyRun{
              t_.slabs[c.layer][segment.name] + slot * t_.row_bytes[segment.name] + segment.dst + piece[i].lo,
              base + segment.src + piece[i].lo, piece[i].hi - piece[i].lo};
          bytes += runs[i].bytes;
        }
        r.dispatched |= bit;
        if (bytes == 0) {
          publish_collected(s, j);
          continue;
        }
        PackJob& job = jobs_[job_index];
        if (!job.done()) throw std::runtime_error("exl3 RAM miss: a packing job was re-armed while a worker still holds it");
        job.arm(runs, segments, pack_split_, fault_.pack_delay_ns, c.trace ? &worker_stamp : nullptr, c.trace);
        pool_->post(&job);  // throws before queueing
        ++c.packing;
        any = true;
      }
    }
    return any;
  }

  // Piece streaming, the owner's publish (plan §3.4 H1): piece j of the row in `slot` is stored and fenced (its job
  // read done, or it had no bytes), so set its bit on every readiness word naming the row. A word that refuses
  // (another generation, or the bit already set) fails the call. The bit is marked published either way: the piece
  // was collected, and nothing else may wait on it.
  void publish_collected(size_t slot, int j) {
    Call& c = c_;
    BounceRow& r = rows_[slot];
    const uint8_t bit = static_cast<uint8_t>(1u << j);
    const bool twice = ++publishes_ == fault_.publish_twice;  // fault: publish_twice is 0 when off
    if (++c.published == c.total * kPieces && fault_.last_publish_delay_ns > 0) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(fault_.last_publish_delay_ns));
    }
    if (c.publish != nullptr && c.publish->rows != nullptr) {
      const PieceTarget& target = c.publish->rows[r.ordinal];
      for (int w = 0; w < target.count; ++w) {
        for (int attempt = 0; attempt < (twice ? 2 : 1); ++attempt) {
          if (publish_piece(target.words[w], c.publish->generation, bit)) continue;
          ++publish_refused_;
          if (c.trace) ++c.trace->piece_publish_refused;
          c.failed = true;
        }
      }
    }
    const int64_t seq = ++c.events;
    if (c.trace) {
      ++c.trace->pieces_published;
      if ((r.published >> (j + 1)) != 0) ++c.trace->pieces_out_of_order;  // a higher-numbered piece went first
      if (r.ordinal < static_cast<size_t>(kTraceRows)) c.trace->piece_publish[r.ordinal][j] = seq;
    }
    r.published |= bit;
  }

  // Piece streaming: collect every piece whose job is done (acquire), publish it, and finish each row whose pieces
  // are all published and whose sub-reads have all retired. Runs every turn, packing or not: a row whose last
  // sub-read retires after its pieces were published (one no piece depends on) is finished here too.
  void collect_pieces() {
    Call& c = c_;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      BounceRow& r = rows_[s];
      if (r.state == RowState::Free) continue;
      const uint8_t held = r.dispatched & static_cast<uint8_t>(~r.published);
      for (int j = 0; j < kPieces && held != 0; ++j) {
        if ((held >> j & 1u) == 0) continue;
        const PackJob& job = jobs_[s * kPieces + static_cast<size_t>(j)];
        if (!job.done()) continue;
        if (j >= 1 && holding_for_probe()) continue;
        r.pack_first = std::min(r.pack_first, job.first_start.load(std::memory_order_relaxed));
        r.pack_last = std::max(r.pack_last, job.last_end.load(std::memory_order_relaxed));
        --c.packing;
        publish_collected(s, j);
      }
      if (r.state == RowState::Ready && r.published == kAllPieces) {
        finish_row(s, r.pack_first == INT64_MAX ? 0 : r.pack_first, r.pack_last);
      }
    }
  }

  // The hold_until_probe_ms fault: true while the request's StreamProbe does not yet read tagged(1, generation) and
  // the hold has not timed out. A failing read is never held: quiesce() must collect every piece.
  bool holding_for_probe() const {
    const Call& c = c_;
    if (c.hold_until == 0 || c.failed || now_ns() >= c.hold_until) return false;
    const uint64_t want = (uint64_t{1} << 56) | (c.publish->generation & ((uint64_t{1} << 56) - 1));
    return __atomic_load_n(c.publish->probe, __ATOMIC_ACQUIRE) != want;
  }

  // Finish every row whose copy the workers have completed: the bank's packing reference is released
  // here, on the owner, and only after its job reads done.
  void collect_packed() {
    Call& c = c_;
    if (piece_stream_) return collect_pieces();
    if (c.packing == 0) return;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (rows_[s].state != RowState::Packing || !jobs_[s].done()) continue;
      finish_row(s, jobs_[s].first_start.load(std::memory_order_relaxed), jobs_[s].last_end.load(std::memory_order_relaxed));
      --c.packing;
    }
  }

  // Wait for every copy the workers still hold, and finish those rows like any other: they were copied
  // whole, and the accounting says so. On a failure the caller still releases every slot; this
  // guarantees nothing writes into them, or reads the bounce, afterwards.
  // With piece streaming every piece job is waited for, then collected and published like any other.
  void quiesce() {
    Call& c = c_;
    if (c.packing == 0) return;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots) && piece_stream_; ++s) {
      const uint8_t held = rows_[s].dispatched & static_cast<uint8_t>(~rows_[s].published);
      for (int j = 0; j < kPieces; ++j) {
        if (held >> j & 1u) {
          while (!jobs_[s * kPieces + static_cast<size_t>(j)].done()) _mm_pause();
        }
      }
    }
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (rows_[s].state != RowState::Packing) continue;
      while (!jobs_[s].done()) _mm_pause();
    }
    collect_packed();
  }

  // The row is packed whole: account it, flag it and free its slot. Packing is the last reference the bank
  // held on this slot: only now may it be reused.
  void finish_row(size_t best, int64_t start, int64_t end) {
    Call& c = c_;
    const size_t ordinal = rows_[best].ordinal;
    if (c.trace) {
      if (ordinal < static_cast<size_t>(kTraceRows)) {
        c.trace->row_pack_start[ordinal] = start;
        c.trace->row_pack_end[ordinal] = end;
      } else {
        ++c.trace->rows_untraced;
      }
      for (const Segment& segment : t_.segments) c.trace->useful_bytes += segment.bytes;
      // Rows pack in completion order and may overlap, so the first to finish is not always the first to start.
      if (c.trace->pack_start == 0 || start < c.trace->pack_start) c.trace->pack_start = start;
      c.trace->pack_end = std::max(c.trace->pack_end, end);
      c.trace->pack_ns += end - start;
    }
    if (c.packed) (*c.packed)[ordinal] = 1;
    if (fault_.poison) std::memset(bounce_slot(best), kPoisonFill ^ 0xFF, static_cast<size_t>(t_.slot_bytes));
    rows_[best] = BounceRow{};
    --rows_busy_[best / kBounceRows];
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
  // Packing workers (set_pack; none by default): a job and a run list per bounce slot, sized at open().
  unsigned pack_workers_ = 0;
  unsigned pack_split_ = 0;
  // Test-only owner-pinning scaffold (set_owner_core; -1 by default, meaning "no pin"). `unpinned_affinity_`
  // is the mask open() found before pinning, restored by the destructor.
  int64_t owner_core_ = -1;
  bool owner_pinned_ = false;
  cpu_set_t unpinned_affinity_{};
  std::unique_ptr<PackPool> pool_;
  // Indexed by bounce slot with the flag off, by (slot, piece) with piece streaming (size_jobs).
  PackJob jobs_[kBounceSlots * kPieces];
  std::vector<CopyRun> runs_;
  // Piece streaming (set_piece_stream; off by default). subs_ is sub-reads per part: 1 with the flag off, which
  // makes descriptor (slot, part, sub) the old (slot, part). sub_reads_ holds each live sub-read's Read (a
  // descriptor points into it), piece_runs_ each slot's piece runs, geometry_ a batch's rows between validation
  // and admission. All sized at open() or set_piece_stream(), and empty with the flag off.
  bool piece_stream_ = false;
  size_t subs_ = 1;
  std::vector<Read> sub_reads_;
  std::vector<PieceRun> piece_runs_;
  RowGeometry geometry_[kBounceRows];
  std::vector<SqeRecord>* sqe_log_ = nullptr;  // test only (set_sqe_log)
  int64_t publishes_ = 0;        // pieces published over the reader's life (the publish_twice fault counts them)
  int64_t publish_refused_ = 0;  // publish attempts a readiness word refused
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
// `owner_core` (test-only owner-pinning scaffold, PACK_WORKERS.md): -1 (the Python wrapper's default)
// leaves the reader byte-for-byte what it is without this parameter; >= 0 pins the calling/owner thread
// to that core and excludes it from the packing pool's mask (RowReader::set_owner_core).
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
    TensorView record,
    int64_t owner_core) {
  using namespace exl3_ram_miss;
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  RowReader reader(
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes),
      direct != 0, f[19], f[20]);
  reader.set_owner_core(owner_core);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) return 0;
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
// (see ReadFault and fault_from), then reads `then_experts` into `then_slots` with no fault (no second read when
// there are none). Results go to
// `results[0..8]`: the two reads' results, the completions the reader had reaped after each, then its
// stale completions, generation wraps, the packing jobs still open when the first read returned, the
// number of packing workers the reader has, and whether its pool was still active (spinning) then.
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
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  RowReader reader(
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes),
      direct != 0, f[19], f[20]);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) {
    out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = out[8] = 0;
    return;
  }
  reader.set_fault(fault_from(f));
  const size_t step = f[18] > 0 ? static_cast<size_t>(f[18]) : static_cast<size_t>(kBounceRows);
  out[0] = reader.read(row, ids_of(experts), slots_of(slots), step, abandon_after(f[17]));
  out[2] = reader.cqes();
  out[4] = reader.stale_cqes();
  out[5] = reader.generation_wraps();
  out[6] = reader.unfinished_jobs();
  out[7] = reader.pack_workers();
  out[8] = reader.pack_pool_active() ? 1 : 0;
  reader.set_fault(ReadFault{});
  if (then_experts.size(0) == 0) return;  // a test that only wants the first read's state
  out[1] = reader.read(row, ids_of(then_experts), slots_of(then_slots), kBounceRows, abandon_after(0));
  out[3] = reader.cqes();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_faulted, exl3_ram_miss_read_rows_faulted);

// Test only (U10): exl3_ram_miss_read_rows_traced's read, recording every SQE the reader prepared. `sqes` receives
// up to sqes.size(0) rows of 4 int64 (file, offset, length, bounce byte offset), in preparation order; `info` 5 int64:
// the result, the SQE count, the descriptor count, the ring credit and the completions reaped. `fault` as the faulted
// call's (word 22 turns piece streaming on).
void exl3_ram_miss_read_rows_sqes(
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
    TensorView record,
    TensorView sqes,
    TensorView info) {
  using namespace exl3_ram_miss;
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  auto* out = static_cast<int64_t*>(info.data_ptr());
  out[0] = out[1] = out[2] = out[3] = out[4] = 0;
  RowReader reader(
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes),
      direct != 0, f[19], f[20]);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) return;
  reader.set_fault(fault_from(f));
  std::vector<RowReader::SqeRecord> log;
  reader.set_sqe_log(&log);
  StageRecord stage;
  const int result = reader.read(
      row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), abandon_after(f[17]), &stage);
  stage.ok = result == 1 ? 1 : 0;
  stage.status = result == 1 ? kStatusServed : result == 0 ? kStatusFailed : kStatusCancelled;
  std::memcpy(record.data_ptr(), &stage, sizeof(stage));
  auto* rows = static_cast<int64_t*>(sqes.data_ptr());
  const size_t kept = std::min<size_t>(log.size(), static_cast<size_t>(sqes.size(0)));
  for (size_t i = 0; i < kept; ++i) {
    rows[4 * i] = log[i].file;
    rows[4 * i + 1] = log[i].offset;
    rows[4 * i + 2] = log[i].length;
    rows[4 * i + 3] = log[i].bounce;
  }
  out[0] = result;
  out[1] = static_cast<int64_t>(log.size());
  out[2] = static_cast<int64_t>(reader.descriptors());
  out[3] = reader.credit();
  out[4] = reader.cqes();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_sqes, exl3_ram_miss_read_rows_sqes);

// Test only (U8): the owner's publish primitive on one readiness word (`word`, one int64): 1 when it set `bit`.
int64_t exl3_ram_miss_publish_piece(TensorView word, int64_t generation, int64_t bit) {
  using namespace exl3_ram_miss;
  return publish_piece(
             static_cast<uint64_t*>(word.data_ptr()), static_cast<uint64_t>(generation), static_cast<uint8_t>(bit))
             ? 1
             : 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_publish_piece, exl3_ram_miss_publish_piece);

// Test only (U2, U3, U6): exl3_ram_miss_read_rows_traced's read, publishing each row's pieces into its readiness
// words: row ordinal o's are masks[o][0 .. masks.size(1)), under `generation` (the caller initialises them). When
// `reference` is not empty (a slab pointer table shaped like `slabs`, holding row o at ref_slots[o]), a checker
// thread polls the first word of every row while the read runs and, for each bit it sees set, compares the piece's
// bytes in the destination slab with the reference: what a device that acquired the bit would copy. `info` 5 int64:
// the result, the reader's refused publishes, the pieces checked, the pieces whose bytes differed, and the bits the
// checker saw set before the read returned.
void exl3_ram_miss_read_rows_pieces(
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
    TensorView record,
    TensorView masks,
    int64_t generation,
    TensorView reference,
    TensorView ref_slots,
    TensorView info) {
  using namespace exl3_ram_miss;
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  auto* out = static_cast<int64_t*>(info.data_ptr());
  std::fill(out, out + 5, 0);
  const Tables t = tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes);
  const std::vector<int32_t> ids = ids_of(experts);
  const std::vector<int64_t> dest = slots_of(slots);
  const size_t lanes = static_cast<size_t>(masks.size(1));
  if (static_cast<size_t>(masks.size(0)) != ids.size() || lanes == 0 || lanes > static_cast<size_t>(kPieceTargets)) {
    throw std::runtime_error("exl3 RAM miss: masks must be [rows, 1..8] readiness words");
  }
  auto* words = static_cast<uint64_t*>(masks.data_ptr());
  std::vector<PieceTarget> targets(ids.size());
  for (size_t o = 0; o < ids.size(); ++o) {
    for (size_t l = 0; l < lanes; ++l) targets[o].words[targets[o].count++] = words + o * lanes + l;
  }
  const PiecePublish publish{static_cast<uint64_t>(generation), targets.data()};
  RowReader reader(Tables(t), direct != 0, f[19], f[20]);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) return;
  reader.set_fault(fault_from(f));

  // The checker: the pieces' runs per row, then poll until the read returns, and once more after.
  const bool checking = reference.numel() > 0;
  const size_t count = t.segments.size();
  std::vector<PieceRun> runs(ids.size() * kPieces * count);
  const auto* ref_table = checking ? static_cast<const int64_t*>(reference.data_ptr()) : nullptr;
  const auto* ref_slot = checking ? static_cast<const int64_t*>(ref_slots.data_ptr()) : nullptr;
  if (checking) {
    for (size_t o = 0; o < ids.size(); ++o) {
      RowGeometry g;
      if (!row_geometry(t, static_cast<size_t>(row * t.experts + ids[o]), g, &runs[o * kPieces * count])) {
        throw std::runtime_error("exl3 RAM miss: the checker cannot cut a row");
      }
    }
  }
  std::vector<uint8_t> seen(ids.size(), 0);
  int64_t checked = 0, differed = 0, early = 0;
  const auto check_pass = [&](bool during) {
    for (size_t o = 0; o < ids.size(); ++o) {
      const uint64_t word = __atomic_load_n(words + o * lanes, __ATOMIC_ACQUIRE);
      if ((word >> 8) != (static_cast<uint64_t>(generation) & ((uint64_t{1} << 56) - 1))) continue;
      const uint8_t fresh = static_cast<uint8_t>(word & 0xFF) & static_cast<uint8_t>(~seen[o]);
      for (int j = 0; j < kPieces; ++j) {
        if ((fresh >> j & 1u) == 0) continue;
        bool same = true;
        for (size_t i = 0; i < count; ++i) {
          const Segment& s = t.segments[i];
          const PieceRun& run = runs[(o * kPieces + static_cast<size_t>(j)) * count + i];
          if (run.lo >= run.hi) continue;
          const uint8_t* got = t.slabs[row][s.name] + dest[o] * t.row_bytes[s.name] + s.dst + run.lo;
          const auto* ref_base = reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(
              ref_table[row * static_cast<int64_t>(t.slabs[row].size()) + s.name]));
          const uint8_t* want = ref_base + ref_slot[o] * t.row_bytes[s.name] + s.dst + run.lo;
          same = same && std::memcmp(got, want, static_cast<size_t>(run.hi - run.lo)) == 0;
        }
        ++checked;
        if (!same) ++differed;
        if (during) ++early;
      }
      seen[o] |= fresh;
    }
  };
  std::atomic<bool> reading{true};
  std::thread checker;
  if (checking) {
    checker = std::thread([&] {
      while (reading.load(std::memory_order_acquire)) check_pass(true);
    });
  }
  StageRecord stage;
  int result = 0;
  try {
    result = reader.read(
        row, ids, dest, static_cast<size_t>(step), abandon_after(f[17]), &stage, nullptr, SIZE_MAX, nullptr, &publish);
  } catch (...) {
    reading.store(false, std::memory_order_release);
    if (checker.joinable()) checker.join();
    throw;
  }
  reading.store(false, std::memory_order_release);
  if (checker.joinable()) checker.join();
  if (checking) check_pass(false);
  stage.ok = result == 1 ? 1 : 0;
  stage.status = result == 1 ? kStatusServed : result == 0 ? kStatusFailed : kStatusCancelled;
  std::memcpy(record.data_ptr(), &stage, sizeof(stage));
  out[0] = result;
  out[1] = reader.publish_refused();
  out[2] = checked;
  out[3] = differed;
  out[4] = early;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_pieces, exl3_ram_miss_read_rows_pieces);

// Test only (U1): the sub-reads and pieces the reader computes when it admits expert `expert` of streamed row `row`
// (row_geometry). `subs`: kPieces rows of 6 int64 (file, offset, length, dest, part, k), in file order; `pieces`:
// kPieces rows of 1 + 2 * segments int64: the dependency mask, then (dst_lo, dst_hi) per segment in segment
// destination coordinates (dst + the run's bounds). Returns the sub-read count, or -1 when the row cannot be cut.
int64_t exl3_ram_miss_piece_geometry(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    std::string source_paths,
    int64_t slot_bytes,
    int64_t row,
    int64_t expert,
    TensorView subs,
    TensorView pieces) {
  using namespace exl3_ram_miss;
  const Tables t = tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes);
  const size_t count = t.segments.size();
  RowGeometry g;
  std::vector<PieceRun> runs(static_cast<size_t>(kPieces) * count);
  if (!row_geometry(t, static_cast<size_t>(row * t.experts + expert), g, runs.data())) return -1;
  auto* sub_out = static_cast<int64_t*>(subs.data_ptr());
  for (int s = 0; s < g.subs; ++s) {
    const int64_t words[6] = {g.sub[s].file, g.sub[s].offset, g.sub[s].length, g.sub[s].dest, g.part[s], g.k[s]};
    std::copy(words, words + 6, sub_out + 6 * s);
  }
  auto* piece_out = static_cast<int64_t*>(pieces.data_ptr());
  const size_t width = 1 + 2 * count;
  for (int j = 0; j < kPieces; ++j) {
    int64_t* line = piece_out + static_cast<size_t>(j) * width;
    line[0] = g.deps[j];
    for (size_t i = 0; i < count; ++i) {
      const PieceRun& run = runs[static_cast<size_t>(j) * count + i];
      line[1 + 2 * i] = t.segments[i].dst + run.lo;
      line[2 + 2 * i] = t.segments[i].dst + run.hi;
    }
  }
  return g.subs;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_piece_geometry, exl3_ram_miss_piece_geometry);

// The stream kernel's piece table (piece-streaming plan 4.2, open question 6: one entry per (row, expert), computed by
// the reader's own row_geometry so the device cannot disagree with it). `runs`: int32 [layers, experts, kPieces,
// segments, 2], each run as (dst_lo, dst_hi) byte offsets into the segment's name row. A row the reader refuses to
// cut gets empty runs; its read fails, so no device copy ever uses them. Returns how many rows were refused.
int64_t exl3_ram_miss_piece_runs(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    std::string source_paths,
    int64_t slot_bytes,
    TensorView runs) {
  using namespace exl3_ram_miss;
  const Tables t = tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes);
  const size_t count = t.segments.size();
  const size_t rows = static_cast<size_t>(t.layers * t.experts);
  if (runs.size(0) != t.layers || runs.size(1) != t.experts || runs.size(2) != kPieces ||
      runs.size(3) != static_cast<int64_t>(count) || runs.size(4) != 2) {
    throw std::runtime_error("exl3 RAM miss: the piece-run table has the wrong shape");
  }
  auto* out = static_cast<int32_t*>(runs.data_ptr());
  std::vector<PieceRun> piece(static_cast<size_t>(kPieces) * count);
  int64_t refused = 0;
  for (size_t row = 0; row < rows; ++row) {
    RowGeometry g;
    int32_t* line = out + row * kPieces * count * 2;
    if (!row_geometry(t, row, g, piece.data())) {
      std::fill(line, line + kPieces * count * 2, 0);
      ++refused;
      continue;
    }
    for (size_t k = 0; k < static_cast<size_t>(kPieces) * count; ++k) {
      const Segment& segment = t.segments[k % count];
      if (segment.dst + piece[k].hi > INT32_MAX) {
        throw std::runtime_error("exl3 RAM miss: a piece run ends past the int32 range of the stream kernel's table");
      }
      line[2 * k] = static_cast<int32_t>(segment.dst + piece[k].lo);
      line[2 * k + 1] = static_cast<int32_t>(segment.dst + piece[k].hi);
    }
  }
  return refused;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_piece_runs, exl3_ram_miss_piece_runs);

// Test only: build a packing pool as if the creating thread could run on the cores set in `inherited`
// (two int64 words, cores 0-127) and write each worker's affinity, as the kernel reports it, to `out`
// (two words per worker). Throws, like the pool, when no core is left.
void exl3_ram_miss_pack_pool_affinity(TensorView inherited, int64_t workers, TensorView out) {
  using namespace exl3_ram_miss;
  const auto* bits = static_cast<const int64_t*>(inherited.data_ptr());
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int core = 0; core < 128; ++core) {
    if ((static_cast<uint64_t>(bits[core / 64]) >> (core % 64)) & 1u) CPU_SET(core, &mask);
  }
  PackPool pool(static_cast<unsigned>(workers), mask, static_cast<size_t>(kBounceSlots));
  auto* words = static_cast<int64_t*>(out.data_ptr());
  for (size_t w = 0; w < pool.workers(); ++w) {
    const cpu_set_t set = pool.worker_affinity(w);
    words[2 * w] = words[2 * w + 1] = 0;
    for (int core = 0; core < 128; ++core) {
      if (CPU_ISSET(core, &set)) words[2 * w + core / 64] |= static_cast<int64_t>(uint64_t{1} << (core % 64));
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pack_pool_affinity, exl3_ram_miss_pack_pool_affinity);

// Test only: the packing pool's switches between parked and spinning workers, on the cores set in `inherited`.
// `out` (6 int64): [0] a post to a pool never made active completes (1/0); [1] the workers' CPU time over 30 ms
// active and idle, ns; [2] a post while active completes; [3] their CPU time over 30 ms after the pool was made
// inactive again (after a 30 ms settle), ns; [4] a post to that cold pool completes; [5] a pool destroyed while
// active joins (1; a pool that did not would hang here). A post "completes" when its job reads done within 5 s.
void exl3_ram_miss_pack_pool_handoff(TensorView inherited, int64_t workers, TensorView out) {
  using namespace exl3_ram_miss;
  const auto* bits = static_cast<const int64_t*>(inherited.data_ptr());
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int core = 0; core < 128; ++core) {
    if ((static_cast<uint64_t>(bits[core / 64]) >> (core % 64)) & 1u) CPU_SET(core, &mask);
  }
  auto* o = static_cast<int64_t*>(out.data_ptr());
  std::vector<uint8_t> src(1 << 16, 7), dst(1 << 16, 0);
  const CopyRun run{dst.data(), src.data(), static_cast<int64_t>(src.size())};
  PackJob job;
  const auto complete = [&](PackPool& pool) {
    job.arm(&run, 1, static_cast<unsigned>(workers), 0, nullptr, nullptr);
    pool.post(&job);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!job.done() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return job.done() ? int64_t{1} : int64_t{0};
  };
  const auto cpu_over = [](PackPool& pool, int ms) {
    const int64_t before = pool.worker_cpu_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    return pool.worker_cpu_ns() - before;
  };
  {
    PackPool pool(static_cast<unsigned>(workers), mask, static_cast<size_t>(kBounceSlots));
    o[0] = complete(pool);
    pool.set_active(true);
    o[1] = cpu_over(pool, 30);
    o[2] = complete(pool);
    pool.set_active(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    o[3] = cpu_over(pool, 30);
    o[4] = complete(pool);
  }
  {
    PackPool pool(static_cast<unsigned>(workers), mask, static_cast<size_t>(kBounceSlots));
    pool.set_active(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  o[5] = 1;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pack_pool_handoff, exl3_ram_miss_pack_pool_handoff);

// Test only: the cores a packing worker may use when the creating thread may use those set in `inherited`
// (two int64 words, cores 0-127), as two words in `out`. Starts no thread.
void exl3_ram_miss_pack_worker_cpus(TensorView inherited, TensorView out) {
  using namespace exl3_ram_miss;
  const auto* bits = static_cast<const int64_t*>(inherited.data_ptr());
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int core = 0; core < 128; ++core) {
    if ((static_cast<uint64_t>(bits[core / 64]) >> (core % 64)) & 1u) CPU_SET(core, &mask);
  }
  const cpu_set_t allowed = pack_worker_cpus(mask);
  auto* words = static_cast<int64_t*>(out.data_ptr());
  words[0] = words[1] = 0;
  for (int core = 0; core < 128; ++core) {
    if (CPU_ISSET(core, &allowed)) words[core / 64] |= static_cast<int64_t>(uint64_t{1} << (core % 64));
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pack_worker_cpus, exl3_ram_miss_pack_worker_cpus);

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
constexpr int64_t kHotHeaderBytes = 8;
constexpr int64_t kHotAlignment = 64;
constexpr uint32_t kHotRecords = kDemandRecords;
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

// ---- Lease block (LEASE_PROTOCOL.md section 4) ----
// The layout is written here, in exl3_ram_miss.cuh and in ops/moe/exl3_lease_block.py; the layout test checks
// that they agree, so these lines use only + - * over integers and known names. The service writes areas H and
// S (header, row table, row results, slot generations); the device writes area D; Python writes nothing.
constexpr int64_t kLeaseRing = 16;  // == kDemandRecords
constexpr int64_t kLeaseLanes = 8;  // == kMaxIds
constexpr int64_t kLeaseHeaderRing = 8;
constexpr int64_t kLeaseHeaderLanes = 12;
constexpr int64_t kLeaseHeaderShutdown = 20;
constexpr int64_t kLeaseHeaderSlotGenOffset = 32;
constexpr int64_t kLeaseHeaderDOffset = 36;
constexpr int64_t kLeaseHeaderPieceOffset = 40;
constexpr int64_t kLeaseRowTable = 128;
constexpr int64_t kLeaseRowResult = 4096;
constexpr int64_t kLeaseRowResultBytes = 32;
constexpr int64_t kLeaseRrReady = 0;
constexpr int64_t kLeaseRrSlotGeneration = 8;
constexpr int64_t kLeaseRrHostSlot = 12;
constexpr int64_t kLeaseRrExpert = 16;
constexpr int64_t kLeaseSlotGen = kLeaseRowResult + kLeaseRing * kLeaseLanes * kLeaseRowResultBytes;
constexpr int64_t kLeaseLaneRequest = 0;
constexpr int64_t kLeaseLaneRequestBytes = 64;
constexpr int64_t kLeaseLrGen = 0;
constexpr int64_t kLeaseLrCount = 8;
constexpr int64_t kLeaseLrRow = 12;
constexpr int64_t kLeaseLrExpert = 16;
constexpr int64_t kLeaseLaneAck = kLeaseLaneRequest + kLeaseRing * kLeaseLaneRequestBytes;
constexpr int64_t kLeaseLaneAckBytes = 8;
constexpr int64_t kLeaseTerminal = kLeaseLaneAck + kLeaseRing * kLeaseLanes * kLeaseLaneAckBytes;
constexpr int64_t kLeaseTerminalBytes = 16;
constexpr int64_t kLeaseTermSkippedMask = 0;
constexpr int64_t kLeaseTermReason = 4;
constexpr int64_t kLeaseTermGen = 8;
// StreamProbe[kLeaseRing], device-written: the stream kernel's tagged(1, generation) once it has copied a piece.
constexpr int64_t kLeaseStreamProbe = kLeaseTerminal + kLeaseRing * kLeaseTerminalBytes;
constexpr int64_t kLeaseStreamProbeBytes = 8;
constexpr int64_t kLeaseRowTableBytes = 8;

// Area P, service-written, at a new header offset (kLeaseHeaderPieceOffset): PieceMask[kLeaseRing][kLeaseLanes],
// a per-lane generation-tagged 8-bit readiness bitmask (piece-streaming plan, LEASE_PROTOCOL.md E1 amendment).
// Each word gets its own 128 B line, so the device's per-lane poll never shares a line with a lane it did not
// ask for. Under piece streaming the tier stores `gen << 8` into each miss lane's word at reservation, and the
// reader's owner sets one bit per packed piece (publish_piece); with the flag off nothing writes it.
constexpr int64_t kLeasePieceMaskLineBytes = 128;
constexpr int64_t kLeasePieceMaskBytes = 8;  // one uint64 per word
constexpr int64_t kLeaseAreaPieceMaskBytes = kLeaseRing * kLeaseLanes * kLeasePieceMaskLineBytes;
static_assert(kPieceTargets >= kLeaseLanes, "a row's pieces are published to at most one word per lane");

// kQuarantine (piece streaming only): a slot whose read failed while a lane still leased it under tag LOADING. Its
// mapping is cleared on entry, it is never taken, evicted or counted as a victim, and it becomes kFree when its last
// lease is retired (retire_leases). A leased slot is never released: that is the S6 rule under piece streaming.
enum : uint8_t { kFree = 0, kLoading = 1, kReady = 2, kQuarantine = 3 };

// RowResult.ready tags (LEASE_PROTOCOL.md 4.3): READY for a lane whose row is resident, LOADING (piece streaming) for
// a miss lane granted at reservation, whose pieces become readable bit by bit through its PieceMask word.
constexpr uint64_t kLeaseTagReady = 1;
constexpr uint64_t kLeaseTagLoading = 2;

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
  kDeferred,  // demands held back because their only victims are leased: one per deferral, none evicted
  kLeasesGranted,     // one per lane of a served request in lease mode
  kLeasesAcked,       // released by the device's acknowledgement
  kLeasesVoided,      // released by a terminal record that named the lane
  kLeaseDoubleSignal, // a lane signalled by both, or twice: released once, counted here
  kLateAfterTerminal, // a request the device had already given up on: dropped without a lease
  kDeferredReuse,     // a demand held back because its request slot still holds an unretired lease row
  // S7. The hit-lane subset of kLeasesGranted: lanes granted BEFORE read() by V1's first phase. Separate because
  // kLeasesGranted cannot distinguish the groups, so a build that publishes nothing early -- falling through to
  // the batched grant -- would satisfy every timing assertion by accident. Zero on the single-phase path.
  kHitLeasesGranted,
  kPieceStreamRefused,   // requests refused because piece streaming is on without two-phase and lease mode
  kPiecePublishRefused,  // piece publishes a readiness word refused, over the reader's life (each failed its read)
  kSlotsQuarantined,     // piece streaming: leased slots of a failed read put in kQuarantine instead of released
  kCounterCount,
};

inline uint32_t load_acquire(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release(uint8_t* address, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(address), value, __ATOMIC_RELEASE);
}

// The lease block's publication words are 64 bits: a tag in the top byte over a 56-bit request generation
// (LEASE_PROTOCOL.md 4.2). Built in code, not in a k-constant: the layout test parses those with + - * only.
inline uint64_t load_acquire64(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint64_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release64(uint8_t* address, uint64_t value) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(address), value, __ATOMIC_RELEASE);
}

inline uint64_t generation_of(uint64_t word) {
  return word & ((uint64_t(1) << 56) - 1);
}

inline uint64_t tag_of(uint64_t word) {
  return word >> 56;
}

inline uint64_t tagged_word(uint64_t tag, uint64_t generation) {
  return (tag << 56) | generation;
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
  std::vector<uint8_t> hot_bitmap;
  // Lease mode: the device's lane list and 56-bit request generation, from the lane request (not the record).
  uint64_t gen = 0;
  std::vector<int32_t> lane_experts;
};

// The service's private account of one request's leases, by request slot (LEASE_PROTOCOL.md 5.2).
struct LaneLease {
  uint8_t state = 0;  // 0 none, 1 granted, 2 acknowledged, 3 voided by a terminal
  int32_t slot = -1;
  uint32_t slot_generation = 0;
  bool counted = false;  // a second signal for this lane was already counted
};

struct Outstanding {
  bool active = false;
  // A further lane group is still to be granted into this entry (V1 two-phase, S1/S4). While it is set the entry
  // counts as open even though no lane is in state 1 yet, so retire_leases cannot free the ring index out from
  // under a grant that has not run.
  bool grants_pending = false;
  uint64_t gen = 0;
  int64_t row = 0;
  uint32_t count = 0;
  LaneLease lane[kLeaseLanes];
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
  std::vector<uint32_t> leases;      // GPU-reader leases per slot (LEASE_PROTOCOL.md section 8); 0 frees a slot for eviction
  std::vector<uint32_t> generation;  // bumped before a slot's bytes change; mirrored into the lease block's SlotGen
  int64_t rows_demand = 0;
  int64_t rows_advisory = 0;
};

// What a request could take from a tier, counted without taking anything.
struct VictimCensus {
  int64_t free = 0;       // FREE slots
  int64_t evictable = 0;  // READY, not hot, not requested, not leased
  int64_t leased = 0;     // as evictable, but leased: they would be victims if the leases retired
};

// The pinned-slot bookkeeping of every streamed layer (plan D12) and the service of one
// request at a time. pump_demand/pump_advice are called by one caller at a time: a test's
// pump(), or the Task 12 thread. The Python-facing methods take the same mutex.
class RamTier {
 public:
  RamTier(
      uint8_t* page, int32_t* slot_map, uint8_t* lease, int64_t lease_bytes, Tables tables, std::vector<int64_t> capacity,
      bool direct, int64_t pack_workers, uint8_t* hot_page, int64_t hot_bytes)
      : page_(page),
        map_(slot_map),
        lease_(lease),
        hot_page_(hot_page),
        layers_(tables.layers),
        experts_(tables.experts),
        reader_(std::move(tables), direct, pack_workers),
        tiers_(static_cast<size_t>(layers_)) {
    hot_stride_ = ((kHotHeaderBytes + (experts_ + 7) / 8 + kHotAlignment - 1) / kHotAlignment) * kHotAlignment;
    if (hot_page_ != nullptr && hot_bytes != kHotRecords * hot_stride_)
      throw std::runtime_error("exl3 RAM miss: hot bitmap sidecar size disagrees with expert count");
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
      tier.leases.assign(tier.capacity, 0);
      tier.generation.assign(tier.capacity, 0);
    }
    if (lease_ != nullptr) init_lease_block(capacity, lease_bytes);
  }

  bool open() {
    if (!reader_.open()) return false;
    next_demand_ = load_acquire(page_ + kDemandDone) + 1u;
    if (next_demand_ == 0) next_demand_ = 1;
    next_advice_ = load_acquire(page_ + kAdviseDone) + 1u;
    if (next_advice_ == 0) next_advice_ = 1;
    return true;
  }

  std::vector<int> packing_cpus() const { return reader_.packing_cpus(); }

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
    retire_leases();  // first, so that an idle pump still retires what the device has acknowledged
    if (admission_closed_.load()) return false;
    const uint32_t head = load_acquire(page_ + kDemandHead);
    if (head == 0 || !reached(head, next_demand_)) return false;
    // A deferred demand is not looked at again until a lease retires: no stage record, no clock read per poll.
    if (deferred_seq_ == next_demand_ && !deferral_may_retry()) return false;
    begin_stage(kStageDemand, next_demand_, head - next_demand_);
    if (head - next_demand_ >= kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq.
      counters_[kOverruns].fetch_add(head - next_demand_ - (kDemandRecords - 2));
      next_demand_ = skip_zero(head - kDemandRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);
    Request request;
    if (read_record(record, next_demand_, &request)) {
      const bool gpu_hot = gpu_hot_mode_.load() && request.armed;
      const bool hot_ok = !gpu_hot ||
          (request.row >= 0 && request.row < layers_ && read_gpu_hot(next_demand_, &request) &&
           load_acquire(record + kRecSeq) == next_demand_);
      if (!hot_ok) {
        counters_[kOverruns].fetch_add(1);
        set_status(record, kFailed);
      } else if (lease_mode_ && request.armed && !read_lane_request(next_demand_, &request)) {
        counters_[kOverruns].fetch_add(1);  // a later request overwrote the lane request: a lapped record
      } else if (lease_mode_ && request.armed && terminal_seen(request)) {
        counters_[kLateAfterTerminal].fetch_add(1);  // the device gave up on it: serve nothing, lease nothing
      } else {
        if (gpu_hot) apply_gpu_hot(request);
        const Defer reason = request.armed ? defers(request) : Defer::kNone;
        if (reason != Defer::kNone) {
          // Held back, not failed and not served: return before handle_demand (so busy_since_ and kBusySeq stay
          // untouched, or the watchdog would count the wait as a hung read) and before the tail (no demand_done, no
          // advance). No stage record is pushed; the first observation time is kept for the one written when it is served.
          if (deferred_seq_ != next_demand_) {
            deferred_seq_ = next_demand_;
            deferred_observed_ns_ = cur_ != nullptr ? cur_->observed : 0;
            counters_[reason == Defer::kRequestSlot ? kDeferredReuse : kDeferred].fetch_add(1);
          }
          deferred_stamp_ = lease_changes_.load();
          deferred_gen_ = request.gen;
          cur_ = nullptr;
          return false;
        } else {
          if (cur_ != nullptr && deferred_seq_ == next_demand_ && deferred_observed_ns_ != 0) {
            cur_->observed = deferred_observed_ns_;
          }
          handle_demand(request, record);
        }
      }
    } else {
      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop
    }
    deferred_seq_ = 0;
    if (const int64_t stall = done_stall_ns_.load(); stall > 0) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(stall));  // test only: see inject_done_stall
    }
    _mm_sfence();
    store_release(page_ + kDemandDone, next_demand_);
    end_stage();
    next_demand_ = skip_zero(next_demand_ + 1u);
    return true;
  }

  // Serve (or skip) the next posted advisory record, if any. True when it handled one.
  bool pump_advice() {
    if (admission_closed_.load()) return false;
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
    bump_generation_locked(row, slot);  // the caller writes the bytes after this returns
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
    if (leased_locked(tiers_[row], slot)) {
      throw std::runtime_error("exl3 RAM miss: release of pinned slot " + std::to_string(slot) + " while it is leased");
    }
    release_locked(row, slot);
    counters_[kVersion].fetch_add(1);
  }

  // Lease mode: the service reads each armed request's lane request, leases every lane's source slot and publishes
  // a row result per lane before it answers (LEASE_PROTOCOL.md 7). Off leaves every request as it always was.
  void set_lease_mode(bool on) {
    if (lease_ == nullptr) throw std::runtime_error("exl3 RAM miss: lease mode needs a lease block");
    if (threaded_.load()) throw std::runtime_error("exl3 RAM miss: set lease mode before the service thread starts");
    lease_mode_ = on;
  }

  void set_gpu_hot(bool on) {
    if (hot_page_ == nullptr) throw std::runtime_error("exl3 RAM miss: GPU hot mode needs a sidecar");
    if (!lease_mode_) throw std::runtime_error("exl3 RAM miss: GPU hot mode needs leases");
    gpu_hot_mode_.store(on);
  }

  bool read_gpu_hot(uint32_t expected, Request* request) const {
    if (hot_page_ == nullptr) return false;
    const uint8_t* record = hot_page_ + static_cast<int64_t>((expected - 1u) % kHotRecords) * hot_stride_;
    if (load_acquire(record) != expected) return false;
    uint32_t count = 0;
    std::memcpy(&count, record + 4, 4);
    if (count != experts_) return false;
    const int64_t bytes = (experts_ + 7) / 8;
    request->hot_bitmap.assign(record + kHotHeaderBytes, record + kHotHeaderBytes + bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire(record) != expected) return false;
    if (experts_ % 8 != 0 &&
        (request->hot_bitmap.back() & static_cast<uint8_t>(~((1u << (experts_ % 8)) - 1u))) != 0)
      return false;
    return true;
  }

  void apply_gpu_hot(const Request& request) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[request.row];
    for (int64_t expert = 0; expert < experts_; ++expert)
      tier.hot[expert] = (request.hot_bitmap[expert / 8] >> (expert % 8)) & 1;
  }

  // Two-phase mode (Task 6 V1): grant the resident lanes inside serve()'s reservation hold, before read(), so the
  // device can copy them while the missing rows are still being read. Off leaves lease mode exactly as Task 5
  // shipped it, which is the A1 arm every Task 6 measurement is reported against.
  void set_two_phase(bool on) {
    if (threaded_.load()) throw std::runtime_error("exl3 RAM miss: set two-phase mode before the service thread starts");
    two_phase_ = on;
  }

  // Piece streaming (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM): the reader reads each part as sub-reads and vets
  // rows piece by piece. Before the thread starts. The reader refuses it without packing workers; the service
  // refuses it without two-phase and lease mode.
  void set_piece_stream(bool on) {
    if (threaded_.load()) throw std::runtime_error("exl3 RAM miss: set piece streaming before the service thread starts");
    reader_.set_piece_stream(on);
  }

  // Shutdown, first step (LEASE_PROTOCOL.md 14.3 S1): the header word tells the device to stop waiting on the service,
  // and the service serves nothing new. Retirement goes on, so acknowledgements of work already in flight still land.
  void close_admission() {
    admission_closed_.store(true);
    if (lease_ != nullptr) store_release(lease_ + kLeaseHeaderShutdown, 1u);
  }

  void inject_done_stall(int64_t ns) {
    done_stall_ns_.store(ns);
  }

  // The seqlock read of the device's lane request for `seq`: false when a later request has already overwritten it.
  bool read_lane_request(uint32_t seq, Request* request) const {
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    const uint8_t* base = lease_ + lease_d_ + kLeaseLaneRequest + idx * kLeaseLaneRequestBytes;
    const uint64_t word = load_acquire64(base + kLeaseLrGen);
    if (tag_of(word) == 0 || (generation_of(word) & 0xFFFFFFFFull) != seq) return false;
    uint32_t count = 0, row = 0;
    std::memcpy(&count, base + kLeaseLrCount, 4);
    std::memcpy(&row, base + kLeaseLrRow, 4);
    int32_t experts[kLeaseLanes];
    std::memcpy(experts, base + kLeaseLrExpert, sizeof(experts));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire64(base + kLeaseLrGen) != word) return false;
    if (count > static_cast<uint32_t>(kLeaseLanes) || static_cast<int64_t>(row) != request->row) return false;
    request->gen = generation_of(word);
    request->lane_experts.assign(experts, experts + count);
    return true;
  }

  // The device has published a terminal for this request: it will never read a source for the lanes it names.
  bool terminal_seen(const Request& request) const {
    return terminal_seen_for(request.seq, request.gen);
  }

  bool terminal_seen_for(uint32_t seq, uint64_t gen) const {
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    const uint8_t* base = lease_ + lease_d_ + kLeaseTerminal + idx * kLeaseTerminalBytes;
    const uint64_t word = load_acquire64(base + kLeaseTermGen);
    return generation_of(word) == gen && tag_of(word) != 0;
  }

  enum class Defer { kNone, kVictims, kRequestSlot };

  // Would this armed demand have to wait for a lease to retire? Counts, changes nothing. The request slot rule (lease
  // mode only): the slot's previous lease row must be fully retired before it is reused. The victim rule: the tier
  // could serve the request only if leased slots were victims (LEASE_PROTOCOL.md section 8, and 20.2c: a dry run,
  // because the take loop evicts a victim per call and keeps that eviction when the request then fails).
  Defer defers(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return Defer::kNone;
    std::lock_guard<std::mutex> guard(mutex_);
    if (lease_mode_ && !request.lane_experts.empty() &&
        outstanding_[static_cast<int64_t>((request.seq - 1u) % kDemandRecords)].active) {
      return Defer::kRequestSlot;
    }
    std::vector<int32_t> wanted;
    for (const auto* ids : {&request.protect, &request.need, &request.lane_experts}) {
      for (int32_t expert : *ids) {
        if (expert >= 0 && expert < experts_ && !listed(wanted, expert)) wanted.push_back(expert);
      }
    }
    const Tier& tier = tiers_[request.row];
    int64_t missing = 0;
    for (int32_t expert : wanted) {
      if (tier.expert_slot[expert] < 0) ++missing;
    }
    if (missing == 0) return Defer::kNone;
    const VictimCensus census = census_locked(request.row, wanted);
    if (census.free + census.evictable < missing && census.free + census.evictable + census.leased >= missing) {
      return Defer::kVictims;
    }
    return Defer::kNone;
  }

  // A deferred demand is retried only when a lease was released since it was last refused, or the device gave up on it.
  bool deferral_may_retry() const {
    if (lease_changes_.load() != deferred_stamp_) return true;
    return lease_mode_ && terminal_seen_for(deferred_seq_, deferred_gen_);
  }

  // S1. Open the ring entry for a request, ONCE, with its full lane count and every lane ungranted.
  //
  // V1 (two-phase) grants a request's lanes in two calls -- the hit lanes inside serve()'s reservation hold, the
  // miss lanes after read() -- so the entry cannot be (re)initialised by the granter: the second call would erase
  // the first call's leases. Opening is therefore its own step. `grants_pending` records that a further grant is
  // still owed, which is what keeps the entry open across the gap (S4); retire_leases must not close a ring slot
  // whose second grant has not run, or grant_lane_group_locked's `entry.active` guard stops protecting it.
  //
  // False on a still-active ring entry, exactly as the single-phase granter was: nothing is opened then.
  //
  // `grants_pending` is false under piece streaming: the loading grant leases every lane, hit and miss, in the one
  // call that follows, so no second grant is ever owed and S4 has nothing to hold open.
  bool open_lease_entry_locked(const Request& request, bool grants_pending = true) {
    if (request.lane_experts.empty()) return true;
    const size_t count = request.lane_experts.size();
    if (count > kLeaseLanes) return false;
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    if (entry.active) return false;  // the request slot still holds an unretired lease row
    entry = Outstanding();
    entry.active = true;
    entry.gen = request.gen;
    entry.row = request.row;
    entry.count = static_cast<uint32_t>(count);
    entry.grants_pending = grants_pending;
    return true;
  }

  // No further grant is owed for this request: the entry may close once its granted lanes retire. Called when the
  // second group has been granted, and on every path that answers without granting it (S5). Without this a
  // request that failed between the two grants would hold its ring index open for the life of the process.
  //
  // An entry that never granted a lane has nothing for the device to retire, so nothing would ever clear
  // `active`. Such an entry is retired here instead; one that does hold leases stays active and is retired by the
  // device's acknowledgement or its terminal, which is the whole of S5.
  void close_pending_grants_locked(const Request& request) {
    if (request.lane_experts.empty()) return;
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    if (!entry.active || entry.gen != request.gen) return;
    entry.grants_pending = false;
    bool held = false;
    for (uint32_t lane = 0; lane < entry.count; ++lane) held = held || entry.lane[lane].state == 1;
    if (!held) entry.active = false;
  }

  // S1. Lease the source slot of every lane `select` picks, and publish its row result. Callable twice over
  // disjoint subsets of one open entry. A lease is counted before its row result is published (6.1).
  //
  // All-or-nothing: every selected lane is validated before any is committed, so a false return has granted
  // nothing and left the entry as it found it.
  //
  // The fence is PER GROUP and must stay that way. It separates THIS call's payload writes from THIS call's ready
  // stores; it deliberately does not cover the other group. That is the correctness core of two-phase: the hit
  // group's ready words have to become visible to the device while the miss group's payloads do not yet exist.
  // Hoisting one fence to cover both groups would either publish hit lanes late (losing the entire mechanism) or
  // publish miss lanes whose payloads have not been written (handing the device a torn row result).
  //
  // The loading grant (piece streaming, plan 3.2): `loading` names this request's newly reserved slots, and a lane
  // whose slot is kLoading and among them is granted too, under tag LOADING rather than READY. Hit lanes still need
  // kReady. Both groups' payloads exist at reservation, so one call, one fence covers them without breaking the rule
  // above: nothing is published whose payload is not written. The caller has stored and fenced each miss lane's
  // PieceMask word (init_piece_words_locked) before this call, so the word carries the request's generation before
  // any ready word of the request is visible. Null `loading` is the two-phase grant exactly as it was.
  template <typename Select>
  bool grant_lane_group_locked(
      const Request& request, Select select, bool hit_phase = false, const std::vector<int64_t>* loading = nullptr) {
    if (request.lane_experts.empty()) return true;
    const size_t count = request.lane_experts.size();
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    if (!entry.active || entry.gen != request.gen || entry.count != count) return false;
    Tier& tier = tiers_[request.row];
    int32_t slots[kLeaseLanes];
    bool take[kLeaseLanes] = {};
    uint64_t tags[kLeaseLanes] = {};
    size_t taken = 0;
    size_t hits = 0;
    for (size_t lane = 0; lane < count; ++lane) {
      if (!select(lane)) continue;
      if (entry.lane[lane].state != 0) return false;  // granted already: the two groups must be disjoint
      const int32_t expert = request.lane_experts[lane];
      if (expert < 0 || expert >= experts_) return false;
      const int32_t slot = tier.expert_slot[expert];
      if (slot < 0 || tier.slot_to_expert[slot] != expert) return false;
      const bool still_loading = loading != nullptr && tier.state[slot] == kLoading &&
                                 std::find(loading->begin(), loading->end(), slot) != loading->end();
      if (tier.state[slot] != kReady && !still_loading) return false;
      slots[lane] = slot;
      take[lane] = true;
      tags[lane] = still_loading ? kLeaseTagLoading : kLeaseTagReady;
      ++taken;
      if (!still_loading) ++hits;
    }
    if (taken == 0) return true;
    uint8_t* results = lease_ + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
    // The writer half of the 11.4 re-read: a ready word is cleared before its payload is rewritten, so a device
    // reader whose payload loads overlap the rewrite finds the word changed when it re-reads it. Without the clear
    // the word would still read the old generation until the store below, and a torn payload would pass.
    for (size_t lane = 0; lane < count; ++lane) {
      if (take[lane]) store_release64(results + lane * kLeaseRowResultBytes + kLeaseRrReady, 0);
    }
    _mm_sfence();
    for (size_t lane = 0; lane < count; ++lane) {
      if (!take[lane]) continue;
      const int32_t slot = slots[lane];
      tier.leases[slot] += 1;
      entry.lane[lane] = LaneLease{1, slot, tier.generation[slot], false};
      uint8_t* result = results + lane * kLeaseRowResultBytes;
      const uint32_t generation = tier.generation[slot];
      const int32_t expert = request.lane_experts[lane];
      const uint16_t row16 = static_cast<uint16_t>(request.row), lane16 = static_cast<uint16_t>(lane);
      std::memcpy(result + kLeaseRrSlotGeneration, &generation, 4);
      std::memcpy(result + kLeaseRrHostSlot, &slot, 4);
      std::memcpy(result + kLeaseRrExpert, &expert, 4);
      std::memcpy(result + 20, &row16, 2);
      std::memcpy(result + 22, &lane16, 2);
    }
    _mm_sfence();  // THIS group's payloads land before THIS group's ready words -- see the note above
    for (size_t lane = 0; lane < count; ++lane) {
      if (!take[lane]) continue;
      store_release64(results + lane * kLeaseRowResultBytes + kLeaseRrReady, tagged_word(tags[lane], request.gen));
    }
    lanes_outstanding_.fetch_add(static_cast<int64_t>(taken));
    counters_[kLeasesGranted].fetch_add(static_cast<int64_t>(taken));
    if (hit_phase) counters_[kHitLeasesGranted].fetch_add(static_cast<int64_t>(hits));  // S7: tag-READY lanes only
    return true;
  }

  // Lease every lane's source slot and publish its row result, in one critical section, after the rows are ready
  // and before the caller answers. A lease is counted before its row result is published (6.1). False on an
  // internal inconsistency: nothing is granted then.
  //
  // The single-phase composition, kept for the non-two-phase path: open the entry and grant every lane at once.
  bool grant_lanes_locked(const Request& request) {
    if (!open_lease_entry_locked(request)) return false;
    const bool granted = grant_lane_group_locked(request, [](size_t) { return true; });
    close_pending_grants_locked(request);  // retires the entry outright when the grant granted nothing
    return granted;
  }

  // Retire the leases the device has acknowledged, or voided with a terminal, without waiting for either. Cheap when
  // nothing is outstanding. Called from the service loop; never blocks (LEASE_PROTOCOL.md 7.5, 16).
  void retire_leases() {
    if (lease_ == nullptr || lanes_outstanding_.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (int64_t idx = 0; idx < kDemandRecords; ++idx) {
      Outstanding& entry = outstanding_[idx];
      if (!entry.active) continue;
      Tier& tier = tiers_[entry.row];
      const uint8_t* acks = lease_ + lease_d_ + kLeaseLaneAck + idx * kLeaseLanes * kLeaseLaneAckBytes;
      const uint8_t* terminal = lease_ + lease_d_ + kLeaseTerminal + idx * kLeaseTerminalBytes;
      const uint64_t stamp = load_acquire64(terminal + kLeaseTermGen);
      const bool terminated = tag_of(stamp) != 0 && generation_of(stamp) == entry.gen;
      uint32_t mask = 0;
      if (terminated) std::memcpy(&mask, terminal + kLeaseTermSkippedMask, 4);
      for (uint32_t lane = 0; lane < entry.count; ++lane) {
        LaneLease& held = entry.lane[lane];
        const uint64_t word = load_acquire64(acks + lane * kLeaseLaneAckBytes);
        const bool acknowledged = tag_of(word) != 0 && generation_of(word) == entry.gen;
        const bool voided = terminated && (mask >> lane & 1u) != 0;
        if (held.state == 1) {
          if (acknowledged) {
            release_lease_locked(tier, held, kLeasesAcked);
          } else if (voided) {
            release_lease_locked(tier, held, kLeasesVoided);
          }
          // Plan 3.3: the last lease of a quarantined slot frees it. Its mapping was cleared on entry, so
          // release_locked unmaps nothing -- the expert may already live in another slot (M3). A kLoading slot only
          // loses the lease here; serve()'s post-read step decides it.
          if (held.state != 1 && tier.state[held.slot] == kQuarantine && !leased_locked(tier, held.slot)) {
            release_locked(entry.row, held.slot);
          }
        }
        // A second signal for a lane already released: counted once, and it releases nothing.
        if (!held.counted && ((held.state == 2 && voided) || (held.state == 3 && acknowledged))) {
          held.counted = true;
          counters_[kLeaseDoubleSignal].fetch_add(1);
        }
      }
      // S4. A lane whose grant has not run yet is still open. Under V1 the miss lanes sit ungranted between the
      // two grants, so counting only state == 1 would clear `active` while a grant is still pending -- after
      // which grant_lane_group_locked's `entry.active` guard no longer protects this ring slot.
      bool open = entry.grants_pending;
      for (uint32_t lane = 0; lane < entry.count; ++lane) open = open || entry.lane[lane].state == 1;
      if (!open) entry.active = false;
    }
  }

  // Lanes leased by the device and not yet retired. An eager pause is refused while this is non-zero; a lease held by
  // anything else (a promotion, in Task 8) is not counted, because it protects its own slot (LEASE_PROTOCOL.md 17.1 R2).
  int64_t graph_leases_outstanding() const {
    return lanes_outstanding_.load();
  }

  // One lease released, exactly once: the per-lane state machine is what makes a second signal harmless.
  void release_lease_locked(Tier& tier, LaneLease& held, int counter) {
    if (tier.leases[held.slot] == 0) {
      throw std::runtime_error("exl3 RAM miss: lease underflow on slot " + std::to_string(held.slot));
    }
    tier.leases[held.slot] -= 1;
    held.state = counter == kLeasesAcked ? 2 : 3;
    lanes_outstanding_.fetch_sub(1);
    counters_[counter].fetch_add(1);
    lease_changes_.fetch_add(1);
  }

  // Test hooks and introspection. slot_info: [state, expert, leases, generation] per slot.
  void slot_info(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      out[4 * slot] = tier.state[slot];
      out[4 * slot + 1] = tier.slot_to_expert[slot];
      out[4 * slot + 2] = tier.leases[slot];
      out[4 * slot + 3] = tier.generation[slot];
    }
  }

  // Test only: the service's account of request slot `idx`: [active, grants_pending, count, gen, then per lane
  // (kLeaseLanes) its state, then per lane its slot].
  void lease_entry(int64_t idx, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Outstanding& entry = outstanding_[idx];
    out[0] = entry.active;
    out[1] = entry.grants_pending;
    out[2] = entry.count;
    out[3] = static_cast<int64_t>(entry.gen);
    for (int64_t lane = 0; lane < kLeaseLanes; ++lane) {
      out[4 + lane] = entry.lane[lane].state;
      out[4 + kLeaseLanes + lane] = entry.lane[lane].slot;
    }
  }

  // Test only: the service grants leases itself from step 3; until then a test stands in for the device's holder.
  void inject_lease(int64_t row, int64_t slot, int64_t delta) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    if (delta < 0 && tier.leases[slot] < static_cast<uint32_t>(-delta)) {
      throw std::runtime_error("exl3 RAM miss: lease underflow on slot " + std::to_string(slot));
    }
    tier.leases[slot] = static_cast<uint32_t>(static_cast<int64_t>(tier.leases[slot]) + delta);
    if (delta < 0) lease_changes_.fetch_add(1);
  }

  VictimCensus victim_census(int64_t row, const std::vector<int32_t>& wanted) {
    std::lock_guard<std::mutex> guard(mutex_);
    return census_locked(row, wanted);
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

  // Test only: carry a whole ReadFault down to this tier's reader, where inject() reaches it only as a
  // delay, a blanket failure or an abandon point. `words` is the reader tests' fault tensor (kFaultWords
  // int64; see fault_from). Unlike fail_reads the fault does NOT short-circuit ahead of the reader: the read
  // runs, so the fault's part errors, pack delay and the rest act on rows that have already packed. The
  // service thread applies it just before its next read (the reader is that thread's alone), and it then
  // stays until replaced; an all-default tensor clears it. Words 17-20 (abandon_after, step, pack_workers,
  // pack_split) and 22 (piece_stream) are not faults and are ignored: use inject() for the abandon point, the
  // tier's own constructor for the packing pool and set_piece_stream() for the mode. The reader's counters (submit and completion calls) run over the
  // reader's whole life, so a call-numbered fault (submit_call, cqe_call) is relative to a fresh tier.
  void inject_fault(const int64_t* words) {
    std::lock_guard<std::mutex> guard(fault_mutex_);
    pending_fault_ = fault_from(words);
    fault_pending_.store(true, std::memory_order_release);
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
    stage_.pack_workers = reader_.pack_workers();
    stage_.pack_split = reader_.pack_split();
    stage_.piece_stream = reader_.piece_stream() ? 1 : 0;
    cur_ = &stage_;
  }

  // Service thread, before a read: install the fault inject_fault() left, on the reader only this thread drives.
  void apply_pending_fault() {
    if (!fault_pending_.load(std::memory_order_acquire)) return;
    ReadFault fault;
    {
      std::lock_guard<std::mutex> guard(fault_mutex_);
      fault = pending_fault_;
      fault_pending_.store(false, std::memory_order_relaxed);
    }
    reader_.set_fault(fault);
  }

  void end_stage() {
    if (cur_ == nullptr) return;
    cur_->done = stamp(cur_);
    last_done_ = cur_->done;
    ring_->push(*cur_);
    cur_ = nullptr;
  }

  static int64_t round_up_page(int64_t value) {
    return (value + 4095) / 4096 * 4096;
  }

  // The service is the only writer of the header, the row table and SlotGen, and writes them before the
  // service thread or any device exists, so plain stores and one fence suffice.
  void init_lease_block(const std::vector<int64_t>& capacity, int64_t lease_bytes) {
    if (reinterpret_cast<uintptr_t>(lease_) % 4096 != 0) {
      throw std::runtime_error("exl3 RAM miss: the lease block must be 4096-byte aligned");
    }
    if (layers_ > (kLeaseRowResult - kLeaseRowTable) / 8) {
      throw std::runtime_error("exl3 RAM miss: too many rows for the lease block's row table");
    }
    slot_gen_base_.assign(static_cast<size_t>(layers_), 0);
    int64_t total_slots = 0;
    for (int64_t row = 0; row < layers_; ++row) {
      slot_gen_base_[row] = total_slots;
      total_slots += capacity[row];
    }
    const int64_t d_offset = round_up_page(kLeaseSlotGen + 4 * total_slots);
    lease_d_ = d_offset;
    const int64_t piece_offset = round_up_page(d_offset + kLeaseStreamProbe + kLeaseRing * kLeaseStreamProbeBytes);
    lease_p_ = piece_offset;
    const int64_t needed = round_up_page(piece_offset + kLeaseAreaPieceMaskBytes);
    if (lease_bytes < needed) {
      throw std::runtime_error(
          "exl3 RAM miss: the lease block has " + std::to_string(lease_bytes) + " bytes, its layout needs " +
          std::to_string(needed));
    }
    auto put_u32 = [&](int64_t offset, uint32_t value) { std::memcpy(lease_ + offset, &value, 4); };
    put_u32(0, 0x4C534531u);  // "LSE1"
    put_u32(4, 2u);           // ABI version (exl3_lease_block.ABI_VERSION; 2 added StreamProbe)
    put_u32(kLeaseHeaderRing, static_cast<uint32_t>(kLeaseRing));
    put_u32(kLeaseHeaderLanes, static_cast<uint32_t>(kLeaseLanes));
    put_u32(16, static_cast<uint32_t>(layers_));
    put_u32(kLeaseHeaderShutdown, 0u);
    put_u32(kLeaseHeaderSlotGenOffset, static_cast<uint32_t>(kLeaseSlotGen));
    put_u32(kLeaseHeaderDOffset, static_cast<uint32_t>(d_offset));
    put_u32(kLeaseHeaderPieceOffset, static_cast<uint32_t>(piece_offset));
    for (int64_t row = 0; row < layers_; ++row) {
      put_u32(kLeaseRowTable + 8 * row, static_cast<uint32_t>(slot_gen_base_[row]));
      put_u32(kLeaseRowTable + 8 * row + 4, static_cast<uint32_t>(capacity[row]));
    }
    slot_gen_ = reinterpret_cast<uint32_t*>(lease_ + kLeaseSlotGen);
    _mm_sfence();
  }

  // A slot is about to hold different bytes: bump its generation, and fence it ahead of the first byte store, so
  // that a GPU reader that re-reads the generation after copying (LEASE_PROTOCOL.md 6.5) sees any rewrite.
  void bump_generation_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const uint32_t next = ++tier.generation[slot];
    if (slot_gen_ != nullptr) {
      store_release(reinterpret_cast<uint8_t*>(slot_gen_ + slot_gen_base_[row] + slot), next);
      _mm_sfence();
    }
  }

  // The eviction predicate's lease half. Task 8 adds host leases here as one more term.
  bool leased_locked(const Tier& tier, int64_t slot) const {
    return tier.leases[slot] > 0;
  }

  // A kQuarantine slot is counted as none of free, evictable or leased: it can never help a deferred request, so it
  // must not make one wait (plan 3.2).
  VictimCensus census_locked(int64_t row, const std::vector<int32_t>& wanted) const {
    const Tier& tier = tiers_[row];
    VictimCensus census;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kFree) {
        ++census.free;
      } else if (tier.state[slot] == kReady) {
        const int32_t expert = tier.slot_to_expert[slot];
        if (tier.hot[expert] || listed(wanted, expert)) continue;
        if (leased_locked(tier, slot)) {
          ++census.leased;
        } else {
          ++census.evictable;
        }
      }
    }
    return census;
  }

  void publish_map(int64_t row, int64_t expert, int32_t slot) {
    __atomic_store_n(map_ + row * experts_ + expert, slot, __ATOMIC_RELEASE);
  }

  // Takes only a kFree slot or evicts an unleased kReady one: kLoading and kQuarantine slots are never taken.
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
      if (leased_locked(tier, slot)) continue;  // a GPU reader may still be reading it
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

  // Plan 3.2: a leased kLoading slot whose read failed. The mapping is cleared at once, both ways, so the expert can
  // be read into another slot by the next request, and the later release_locked (retire_leases, on the last lease)
  // finds no expert to unmap: it cannot unmap the expert from the slot it was re-read into (M3). Nothing is
  // published, because the map was never published for a kLoading slot. The lease is kept: this slot's bytes may be
  // under a device copy until the lane is acknowledged or voided.
  void quarantine_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const int32_t expert = tier.slot_to_expert[slot];
    if (expert >= 0 && tier.expert_slot[expert] == slot) tier.expert_slot[expert] = -1;
    tier.slot_to_expert[slot] = -1;
    tier.state[slot] = kQuarantine;
    counters_[kSlotsQuarantined].fetch_add(1);
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
  bool serve(const Request& request, bool advisory, int64_t* rows, bool* deferred = nullptr) {
    *rows = 0;
    if (deferred != nullptr) *deferred = false;
    if (cur_) cur_->lanes = request.lanes;
    std::vector<int32_t> wanted;
    for (const auto* ids : {&request.protect, &request.need}) {
      for (int32_t expert : *ids) {
        if (!listed(wanted, expert)) wanted.push_back(expert);  // one slot per expert (device bytes may repeat)
      }
    }
    // A lane's own expert must never be a victim of the request that leases it, whatever the post kernel protected.
    for (int32_t expert : request.lane_experts) {
      if (!listed(wanted, expert)) wanted.push_back(expert);
    }
    std::vector<int32_t> missing;
    std::vector<int64_t> slots;
    bool ok = request.row >= 0 && request.row < layers_;
    // Piece streaming publishes into the lease block's readiness words, initialised in the reservation hold below
    // before the two-phase hit grant. Without two-phase and lease mode there is neither, so refuse before any slot
    // is taken (the service refuses the flag too; this is the tier's own guard).
    const bool piece_stream = reader_.piece_stream();
    if (ok && piece_stream && !(two_phase_ && lease_mode_)) {
      counters_[kPieceStreamRefused].fetch_add(1);
      ok = false;
    }
    bool publishing = false;  // piece streaming: this request's miss lanes have readiness words to publish into
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
      if (ok && !missing.empty()) {
        // A request the tier could serve only once leases retire is refused BEFORE any slot is taken: the take
        // loop unmaps a victim per call and keeps that eviction when the request then fails, so a refusal that
        // ran it would evict a row on every retry. A demand is counted as deferred (step 3 retries it); an
        // advisory simply gives up. When leases could not help either, the loop runs as it always has.
        const VictimCensus census = census_locked(request.row, wanted);
        const int64_t want = static_cast<int64_t>(missing.size());
        if (census.free + census.evictable < want && census.free + census.evictable + census.leased >= want) {
          if (!advisory) {
            counters_[kDeferred].fetch_add(1);
            if (deferred != nullptr) *deferred = true;
          }
          ok = false;
        }
      }
      for (size_t i = 0; ok && i < missing.size(); ++i) {
        int64_t evicted = -1;
        const int64_t slot = take_slot_locked(request.row, wanted, false, &evicted);
        if (slot < 0) {
          ok = false;
          break;
        }
        bump_generation_locked(request.row, slot);  // before any byte of the new row is written
        tier.slot_to_expert[slot] = missing[i];
        tier.state[slot] = kLoading;
        tier.expert_slot[missing[i]] = static_cast<int32_t>(slot);
        slots.push_back(slot);
      }
      // S2. Grant and publish the HIT lanes here: in the same mutex_ hold as the reservation, after the take loop
      // has completed with ok still true, and before the hold is dropped for read(). A lane is a hit iff its
      // expert is not in `missing`. This is the whole of V1: these row results become visible to the device while
      // the missing rows are still being read, so their copies overlap the read instead of following it.
      //
      // Neither ordering below it is available. Before the take loop, the !ok bail here returns with no lease
      // unwind, and the deferral branch above sets ok = false for a request that is retried under the same seq --
      // either way leases outlive a request that never ran. After the hold is dropped, the hit slot can be
      // evicted in exactly the window this task exists to close, and the grant buys nothing.
      //
      // Piece streaming grants the MISS lanes here too (plan 3.2): under tag LOADING, into the slots just reserved,
      // in the same all-or-nothing call and behind the same one fence, after init_piece_words_locked has stored and
      // fenced their PieceMask words. No grant is then owed after the read, so the entry opens with none pending.
      if (ok && two_phase_ && lease_mode_ && !advisory && !request.lane_experts.empty()) {
        if (!open_lease_entry_locked(request, !piece_stream)) {
          ok = false;
        } else {
          // Piece streaming: each miss lane's readiness word starts this request's generation with no piece bit,
          // fenced before any ready word of the request. Only after the entry opened: an active entry means an
          // older request may still be reading this ring index's words, and opening refuses it.
          if (piece_stream) publishing = init_piece_words_locked(request, missing);
          if (!(piece_stream ? grant_lane_group_locked(request, [](size_t) { return true; }, true, &slots)
                             : grant_lane_group_locked(
                                   request,
                                   [&](size_t lane) { return !listed(missing, request.lane_experts[lane]); },
                                   true))) {
            close_pending_grants_locked(request);  // granted nothing: retire the entry rather than leak the ring slot
            ok = false;
          }
        }
      }
      if (!ok) {
        for (int64_t slot : slots) {
          // S6. §2's first fact as an assertion rather than an argument: `slots` holds only newly taken slots for
          // experts in `missing`, and a hit lane's expert is by definition not in `missing`, so no slot released
          // here is ever leased. release_locked checks nothing, and a leased slot through it is silent
          // corruption -- the next request is handed a slot the GPU may still be reading. Piece streaming leases
          // miss slots in this hold, but only through the grant above, which is all-or-nothing and the last step
          // that can set ok = false: a request that reaches this branch granted nothing, so the rule holds as is.
          assert(!leased_locked(tiers_[request.row], slot));
          release_locked(request.row, slot);
        }
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
      apply_pending_fault();
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
            advisory ? 1 : SIZE_MAX,
            // Safety, not just style: read() runs here with mutex_ NOT held -- the only lock_guard in
            // serve() near it is the scoped S2 reservation hold above, which closes before this call.
            // retire_leases() takes mutex_ itself, so this is deadlock-free today. That is a property
            // of the code as it stands, not a guarantee: if a future yield point is ever added to
            // read()'s drain loop under a lock, this callback would deadlock against it.
            [this] { retire_leases(); },
            publishing ? &piece_publish_ : nullptr);
        if (piece_stream) counters_[kPiecePublishRefused].store(reader_.publish_refused());
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
          } else if (piece_stream && leased_locked(tier, slots[i])) {
            // Plan 3.3: a miss lane still leases this slot under tag LOADING, and the device may have copied
            // (or be copying) its published pieces. It is quarantined, never released; its last lease frees it.
            quarantine_locked(request.row, slots[i]);
          } else {
            // S6 on the post-read path. With the flag off no miss slot is leased before the S3 grant below. Under
            // piece streaming a leased slot took the branch above, so what reaches here is unleased: its lanes
            // were already voided (a timeout while reading) and nothing can read it any more.
            assert(!leased_locked(tier, slots[i]));
            release_locked(request.row, slots[i]);
          }
        }
        (advisory ? tier.rows_advisory : tier.rows_demand) += published;
        counters_[kVersion].fetch_add(1);
      }
    }
    if (lease_mode_ && !advisory) {
      // Every lane's source is leased, and its row result published, before the caller answers the request.
      //
      // S3. Under two-phase this grants the MISS lanes only, into the entry S2 opened; it must not reopen that
      // entry, which would erase the hit leases S2 already recorded and their row results with them.
      //
      // S5. The failure paths below now hold hit leases where Task 5 held none, and they deliberately do nothing
      // about them: fail_reads, read() returning 0, a cancel (result == -1, reachable only on the advisory path
      // today -- it becomes reachable for demands if the cancel predicate ever widens), and this grant itself
      // failing. In every one the hit leases STAY OUTSTANDING and the request answers with a non-kServed status.
      // They are retired by the device's stage-1 acknowledgement or voided by its terminal. Releasing them on the
      // host is the silent-corruption path: release_locked checks no lease and take_slot_locked's free-slot loop
      // consults none, so the slot goes straight to the next request while the GPU may still be reading it.
      //
      // Piece streaming has no S3: the miss lanes were granted under tag LOADING at reservation, and their readiness
      // is the PieceMask words plus the request's status, not a second grant.
      std::lock_guard<std::mutex> guard(mutex_);
      if (two_phase_) {
        if (ok && !piece_stream) {
          if (!grant_lane_group_locked(request, [&](size_t lane) { return listed(missing, request.lane_experts[lane]); }))
            ok = false;
        }
        close_pending_grants_locked(request);  // the second grant is owed no longer, however this request answered
      } else if (ok) {
        if (!grant_lanes_locked(request)) ok = false;
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

  // Store piece_word(gen) into the readiness word of every lane whose expert is read (in `missing`), then fence,
  // and record those words as the owner's publish targets, by the row's ordinal in the read. Service thread only:
  // it is the only writer of area P. False when no lane names a missing row (nothing to publish).
  bool init_piece_words_locked(const Request& request, const std::vector<int32_t>& missing) {
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    piece_targets_.assign(missing.size(), PieceTarget{});
    bool any = false;
    for (size_t lane = 0; lane < request.lane_experts.size() && lane < static_cast<size_t>(kLeaseLanes); ++lane) {
      const auto found = std::find(missing.begin(), missing.end(), request.lane_experts[lane]);
      if (found == missing.end()) continue;
      uint8_t* word = lease_ + lease_p_ + (idx * kLeaseLanes + static_cast<int64_t>(lane)) * kLeasePieceMaskLineBytes;
      store_release64(word, piece_word(request.gen));
      PieceTarget& target = piece_targets_[static_cast<size_t>(found - missing.begin())];
      target.words[target.count++] = reinterpret_cast<uint64_t*>(word);
      any = true;
    }
    _mm_sfence();
    piece_publish_ = PiecePublish{
        request.gen, piece_targets_.data(),
        reinterpret_cast<const uint64_t*>(lease_ + lease_d_ + kLeaseStreamProbe + idx * kLeaseStreamProbeBytes)};
    return any;
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
  uint8_t* lease_;             // the lease block, or null when the service runs without one
  uint8_t* hot_page_ = nullptr;
  int64_t hot_stride_ = 0;
  std::atomic<bool> gpu_hot_mode_{false};
  uint32_t* slot_gen_ = nullptr;  // SlotGen[] inside it
  std::vector<int64_t> slot_gen_base_;  // first SlotGen word of each row
  int64_t lease_d_ = 0;                 // byte offset of area D (the device-written words)
  int64_t lease_p_ = 0;                 // byte offset of area P (the piece readiness words)
  // Piece streaming: serve()'s readiness words per row it reads, reused every request (like packed_).
  std::vector<PieceTarget> piece_targets_;
  PiecePublish piece_publish_;
  bool lease_mode_ = false;             // set before the service thread starts; off is today's protocol
  bool two_phase_ = false;              // Task 6 V1: hit lanes granted before read(); off is the Task 5 batched grant
  Outstanding outstanding_[kDemandRecords];  // by request slot; guarded by mutex_
  std::atomic<int64_t> lanes_outstanding_{0};  // lanes GRANTED and not yet retired: an early-out for retire_leases
  std::atomic<bool> admission_closed_{false};  // shutdown: serve nothing new; retirement continues
  std::atomic<int64_t> done_stall_ns_{0};      // test only: sleep between serving a demand and storing demand_done
  std::atomic<uint64_t> lease_changes_{0};     // bumped whenever a lease is released: what wakes a deferred demand
  // The demand held back, if any (service thread only): its sequence, the changes seen when it was last refused,
  // its generation (a terminal for it also wakes it) and when it was first observed (for the stage record).
  uint32_t deferred_seq_ = 0;
  uint64_t deferred_stamp_ = 0;
  uint64_t deferred_gen_ = 0;
  int64_t deferred_observed_ns_ = 0;
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
  std::mutex fault_mutex_;  // guards pending_fault_ between inject_fault() and the service thread
  ReadFault pending_fault_{};
  std::atomic<bool> fault_pending_{false};
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
    int64_t direct,
    TensorView lease,
    int64_t pack_workers,
    TensorView hot_page) {
  using namespace exl3_ram_miss;
  const auto* capacity_data = static_cast<const int64_t*>(capacity.data_ptr());
  auto tier = std::make_shared<RamTier>(
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(slot_map.data_ptr()),
      static_cast<uint8_t*>(lease.data_ptr()),
      lease.size(0),
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes),
      std::vector<int64_t>(capacity_data, capacity_data + capacity.size(0)),
      direct != 0,
      pack_workers,
      hot_page.size(0) ? static_cast<uint8_t*>(hot_page.data_ptr()) : nullptr,
      hot_page.size(0));
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

void exl3_ram_miss_slot_info(int64_t handle, int64_t row, TensorView out) {
  exl3_ram_miss::find(handle)->slot_info(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_lease_entry(int64_t handle, int64_t idx, TensorView out) {
  if (idx < 0 || idx >= exl3_ram_miss::kDemandRecords) throw std::runtime_error("exl3 RAM miss: request slot out of range");
  exl3_ram_miss::find(handle)->lease_entry(idx, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_inject_lease(int64_t handle, int64_t row, int64_t slot, int64_t delta) {
  exl3_ram_miss::find(handle)->inject_lease(row, slot, delta);
}

// out: free, evictable, leased.
void exl3_ram_miss_victim_census(int64_t handle, int64_t row, TensorView wanted, TensorView out) {
  const auto census = exl3_ram_miss::find(handle)->victim_census(row, exl3_ram_miss::ids_of(wanted));
  auto* result = static_cast<int64_t*>(out.data_ptr());
  result[0] = census.free;
  result[1] = census.evictable;
  result[2] = census.leased;
}

int64_t exl3_ram_miss_busy_since(int64_t handle) {
  return exl3_ram_miss::find(handle)->busy_since();
}

void exl3_ram_miss_close_admission(int64_t handle) {
  exl3_ram_miss::find(handle)->close_admission();
}

void exl3_ram_miss_set_lease_mode(int64_t handle, int64_t on) {
  exl3_ram_miss::find(handle)->set_lease_mode(on != 0);
}

void exl3_ram_miss_set_gpu_hot(int64_t handle, int64_t on) {
  exl3_ram_miss::find(handle)->set_gpu_hot(on != 0);
}

void exl3_ram_miss_set_two_phase(int64_t handle, int64_t on) {
  exl3_ram_miss::find(handle)->set_two_phase(on != 0);
}

void exl3_ram_miss_set_piece_stream(int64_t handle, int64_t on) {
  exl3_ram_miss::find(handle)->set_piece_stream(on != 0);
}

void exl3_ram_miss_inject_done_stall(int64_t handle, int64_t ns) {
  exl3_ram_miss::find(handle)->inject_done_stall(ns);
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

// Test only: a full ReadFault for the tier's reader (the reader tests' fault tensor; see RamTier::inject_fault).
void exl3_ram_miss_inject_fault(int64_t handle, TensorView fault) {
  exl3_ram_miss::check_fault_words(fault);
  exl3_ram_miss::find(handle)->inject_fault(static_cast<const int64_t*>(fault.data_ptr()));
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_slot_info, exl3_ram_miss_slot_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject_lease, exl3_ram_miss_inject_lease);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_lease_entry, exl3_ram_miss_lease_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_victim_census, exl3_ram_miss_victim_census);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_busy_since, exl3_ram_miss_busy_since);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_close_admission, exl3_ram_miss_close_admission);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_lease_mode, exl3_ram_miss_set_lease_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_gpu_hot, exl3_ram_miss_set_gpu_hot);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_two_phase, exl3_ram_miss_set_two_phase);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_piece_stream, exl3_ram_miss_set_piece_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject_done_stall, exl3_ram_miss_inject_done_stall);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_mapping, exl3_ram_miss_mapping);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_slot_to_expert, exl3_ram_miss_slot_to_expert);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_lru_order, exl3_ram_miss_lru_order);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_hot, exl3_ram_miss_set_hot);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject, exl3_ram_miss_inject);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject_fault, exl3_ram_miss_inject_fault);
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

  // 1 paused, 0 timed out, 2 refused: a graph lane still holds a lease after one retirement pass (the caller must
  // have synchronized the stream, so the device's acknowledgements are visible), and the slots are not the caller's.
  int pause(int64_t timeout_ns) {
    tier_->request_pause(true);
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(true);
    const int64_t deadline = now_ns() + timeout_ns;
    while (!paused_.load()) {
      if (now_ns() > deadline) {
        resume();
        return 0;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    tier_->retire_leases();
    if (tier_->graph_leases_outstanding() > 0) {
      resume();
      return 2;
    }
    return 1;
  }

  // Advisories posted while paused predate the eager use: skip them too.
  void resume() {
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(false);
    tier_->request_pause(false);
  }

 private:
  void run() {
    pthread_setname_np(pthread_self(), "exl3-ram-miss");
    int error = 0;
    if (cpu_core_ >= 0) {
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      CPU_SET(cpu_core_, &cpus);
      error = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    } else {
      // Unpinned, it still keeps off the packing workers' CPUs: they spin at 100% while a read is in service, and
      // this is the thread that feeds them. Left alone, CFS may time-slice it against one of them for milliseconds.
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      if (pthread_getaffinity_np(pthread_self(), sizeof(cpus), &cpus) == 0) {
        for (int cpu : tier_->packing_cpus()) CPU_CLR(cpu, &cpus);
        // Best effort: a narrowing that fails leaves the thread as it was, which is how it ran before.
        if (CPU_COUNT(&cpus) > 0) pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
      }
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
  return exl3_ram_miss::find_thread(handle)->pause(timeout_ns);
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
