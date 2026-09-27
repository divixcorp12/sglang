// Test-only fault injection for RowReader (expert_stream_read_rows_faulted).
#pragma once

#include "row_tables.h"

namespace sglang {
namespace expert_stream {

// Test-only fault injection for RowReader (expert_stream_read_rows_faulted).
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
// last_publish_delay_ns. Keep the layout in step with _fault_tensor in ops/moe/expert_stream_transport.py.
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

template <ExpertRowLayout Layout>
inline void check_fault_words(TensorView fault) {
  if (fault.size(0) != kFaultWords) throw std::runtime_error(error_prefix<Layout>() + "the fault tensor has the wrong length");
}

// Entry points' abandon callback: stop once `after` batches were admitted (0: never).
inline std::function<bool(size_t)> abandon_after(int64_t after) {
  return [after](size_t admitted) { return after > 0 && admitted >= static_cast<size_t>(after); };
}


}  // namespace expert_stream
}  // namespace sglang
