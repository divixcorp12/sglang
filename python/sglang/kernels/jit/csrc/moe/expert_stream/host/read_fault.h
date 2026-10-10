// Test-only fault injection for the expert-stream readers (ReaderCore, expert_stream_read_rows_faulted).
//
// The test entry points take a fault as an int64 tensor of kFaultWords words, whose layout mirrors _fault_tensor in
// ops/moe/expert_stream_transport.py. This file holds the decoded form and the decoding.
//
//   ReadFault        the faults one reader can inject, and the knobs that narrow them
//   kFaultWords      the wire layout of the fault tensor
//   fault_from       tensor words -> ReadFault
//   injects_fault    whether a tensor arms any fault (a production host refuses those)
//   abandon_after    the entry points' abandon callback
//
// Only InstrBuild readers carry a ReadFault; ProdBuild has no fault state at all.
#pragma once

#include "row_tables.h"

namespace sglang {
namespace expert_stream {

// The faults a ReaderCore can inject, and the fields that narrow where they land.
//
// Call numbers (`submit_call`, `cqe_call`, `stale_cqe_call`, `publish_twice`, `submit_short_call`) are 1-based and
// count over the reader's life. "Armed" fields start a fault; "narrowing" fields (part, ordinal, sub, leg, ...) only
// choose which completion it hits and arm nothing alone.
//
// Submit and completion faults:
//   - submit_error / submit_call / submit_first: the submit_call-th submit returns errno submit_error. With
//     submit_first the prepared SQEs are submitted before failing, so reads are in flight.
//   - submit_short_call: that submit consumes nothing and reports success.
//   - cqe_error / cqe_call: replaces the cqe_call-th reaped completion's result with -errno.
//   - ring_reset_fail, nop_flush_refused: with a submit fault that leaves SQEs unconsumed and a refused NOP drain
//     (nop_flush_refused, implied by ring_reset_fail), the failure-path drain's ring reset throws "io_uring ring reset
//     failed" and read() must rethrow it, not abort.
//
// Per-extent faults hit the first completion of an extent of part `part` (-1: none), whichever row it is in and however
// the kernel orders completions. part_error replaces its result with -errno; part_short makes it report at most that
// many bytes (a block multiple). They are narrowed by:
//   - ordinal: the row with that index in the request (-1: any);
//   - sub: the sub-read with that index within its part, piece streaming only (-1: any; without piece streaming every
//     extent is sub 0);
//   - leg: the leg of a fanned-out fixed read (-1: any; default reads are one leg, leg 0). It also narrows cqe_error
//   and
//     hold_ordinal.
//   - short_is_eof: the part_short completion also ends its sub-read, as a file ending there would, so the sub-read
//     retires with fewer bytes than its pieces need (piece streaming only).
//
// Ordering and credit:
//   - reverse_cqes: process each reaped batch back to front. Nothing in the reader may depend on delivery order and the
//     kernel guarantees none across drives, so this makes that requirement testable.
//   - max_outstanding: cap outstanding reads below the ring's depth. Production constants keep a batch (kBounceRows
//     rows) inside the ring, so credit never binds there; this reaches the refill path without resizing the ring.
//   - hold_ordinal / hold_rest: simulate a slow drive. The completions of the row with that index are reaped from the
//     CQ (the kernel is done with them) but withheld from the reader until nothing else is in flight or waiting to
//     pack. Cached buffered reads complete inside submit, so without this no test can hold an extent outstanding while
//     other rows pack. hold_rest withholds every row from that ordinal on and releases them together, so they become
//     ready in one reap. With `sub` set, the hold narrows to that sub-read (of part `part`, or of any part when part is
//     -1), so one sub-read of a row lands last while the row's others land.
//
// Pipeline faults:
//   - pack_delay_ns: sleep inside every row's packing, so the other bank's completions pile up meanwhile.
//   - poison: fill a bounce slot with one pattern when a row takes it and another when the row has packed, and scribble
//     every retired descriptor. A row packed before its reads landed, a bank reused early or a retired descriptor used
//     again then shows in the bytes or crashes.
//   - stale_cqe_call: the k-th extent to retire has its completion delivered again once its descriptor is recycled for
//     another extent.
//   - generation_start: seeds the generation counter (near 2^32 it wraps within a test).
//
// Two-span faults (TwoSpanRows):
//   - suffix_delay_ns: every second-span sub-read's completions are withheld, as hold_ordinal's are, and released once
//     nothing else is in flight and this long after the first was withheld: a slow second span after the first landed.
//
// Piece streaming faults:
//   - publish_twice: the k-th piece published is published a second time, as a re-dispatch would; the readiness word
//     must refuse it.
//   - last_publish_delay_ns: the read's last piece publish sleeps this long first, so it lands just before kDemandDone
//     (device tests).
struct ReadFault {
  int submit_error = 0;
  int64_t submit_call = 0;
  bool submit_first = false;
  int cqe_error = 0;
  int64_t cqe_call = 0;
  int64_t part = -1;
  int part_error = 0;
  int64_t part_short = 0;  // >0: that completion reports at most this many bytes
  bool reverse_cqes = false;
  int64_t max_outstanding = 0;  // >0: cap outstanding reads to this many
  int64_t pack_delay_ns = 0;
  bool poison = false;
  int64_t stale_cqe_call = 0;
  int64_t generation_start = 0;
  int64_t submit_short_call = 0;
  int64_t ordinal = -1;
  int64_t hold_ordinal = -1;
  bool hold_rest = false;
  int64_t sub = -1;
  int64_t publish_twice = 0;
  bool short_is_eof = false;
  int64_t last_publish_delay_ns = 0;
  int64_t leg = -1;
  bool ring_reset_fail = false;
  bool nop_flush_refused = false;
  int64_t suffix_delay_ns = 0;
};

// The fault tensor of the test entry points: kFaultWords int64 words, in the order of _fault_tensor in
// ops/moe/expert_stream_transport.py (keep the two in step).
//
//   0-16   submit_error, submit_call, submit_first, cqe_error, cqe_call, part, part_error, part_short, reverse_cqes,
//          max_outstanding, pack_delay_ns, poison, stale_cqe_call, generation_start, submit_short_call, ordinal,
//          hold_ordinal
//   17     abandon_after (not a reader fault): the entry point's abandon callback says stop once that many batches were
//          admitted (0: never)
//   18     step (not a fault): rows per batch of the faulted call (0: kBounceRows)
//   19     two_span_rows (not a fault): a bit per row ordinal the test entry points read in two spans (TwoSpanRows)
//   20     suffix_delay_ns
//   21     hold_rest
//   22     piece_stream (not a fault): turns piece streaming on before the reader opens
//   23-25  sub, publish_twice, short_is_eof
//   26     reserved
//   27     last_publish_delay_ns
//   28     fixed_chunk_cap (not a fault): caps the registered-buffer chunk size before the reader opens (0: 1 GiB)
//   29     leg
//   30     ring-reset bits: bit 0 nop_flush_refused, bit 1 ring_reset_fail
//   31     leg_cut_cap (not a fault): cuts every read at that many bytes before the reader opens (0: READ_CUTS and the
//          device limits)
constexpr int64_t kFaultWords = 32;

// Decodes the fault words of a tensor of kFaultWords entries.
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
  fault.last_publish_delay_ns = f[27];
  fault.leg = f[29];
  fault.nop_flush_refused = (f[30] & 1) != 0;
  fault.ring_reset_fail = (f[30] & 2) != 0;
  fault.suffix_delay_ns = f[20];
  return fault;
}

// Whether fault words `f` inject a fault, i.e. set a word that arms one. Words that only narrow a fault (the call
// numbers, part, ordinal, sub, leg, submit_first, short_is_eof, hold_rest) arm nothing alone, and words 17-19, 22, 26,
// 28 and 31 are not faults. A production host has no fault state and refuses a tensor for which this is true
// (HostTestExports::install_fault); it needs no ReadFault to decide.
inline bool injects_fault(const int64_t* f) {
  return f[0] != 0 || f[3] != 0 || f[6] != 0 || f[7] != 0 || f[8] != 0 || f[9] > 0 || f[10] > 0 || f[11] != 0 ||
         f[12] > 0 || f[13] != 0 || f[14] != 0 || f[16] >= 0 || f[20] > 0 || f[24] > 0 || f[27] > 0 || f[30] != 0;
}

// Throws if `fault` is not kFaultWords long.
template <ExpertRowLayout Layout>
inline void check_fault_words(TensorView fault) {
  if (fault.size(0) != kFaultWords)
    throw std::runtime_error(error_prefix<Layout>() + "the fault tensor has the wrong length");
}

// The entry points' abandon callback: stops once `after` batches were admitted (0: never).
inline auto abandon_after(int64_t after) {
  return [after](size_t admitted) { return after > 0 && admitted >= static_cast<size_t>(after); };
}

}  // namespace expert_stream
}  // namespace sglang
