// The bounce-bank pool geometry and per-row stage-trace constants shared by every host reader.
#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <immintrin.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
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

// Piece streaming (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM, plan 2026-09-24-dsv41-piece-streaming §4.1-4.3). With
// it on, each nonzero part of a row is read as up to kSubReads page-aligned sub-reads, and the row's needed bytes are
// cut into kPieces pieces: piece j is sub-read j of the row (in file order) mapped into segment destination
// coordinates, its inner cuts rounded down to kPieceAlign. These are fixed, not knobs: the device's readiness word
// carries one bit per piece. With the flag off none of this is used and the reader issues one read per part.
constexpr int kSubReads = 4;
constexpr int kPieces = 8;
constexpr uint8_t kAllPieces = 0xFF;
constexpr int64_t kPieceAlign = 128;
// Row images (direct mode): O_DIRECT's file offset and segment length alignment on the mirror drives (XFS and ext4,
// 512 B logical blocks; exl3_row_image.IO_ALIGN). Slab rows are held to it too, which covers dio_mem_align (4).
constexpr int64_t kImageAlign = 512;
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


}  // namespace exl3_ram_miss
}  // namespace sglang
