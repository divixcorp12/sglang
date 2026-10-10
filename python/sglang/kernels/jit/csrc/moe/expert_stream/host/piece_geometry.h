// Piece streaming geometry: how a row's reads are cut into sub-reads and pieces, and how a piece is published.
//
// With piece streaming a row is not published whole. Each part is read as up to kSubReads sub-reads, the row's bytes
// are cut into kPieces pieces along the sub-read boundaries, and a piece is published (one bit in the lease's
// readiness word) as soon as the sub-reads it depends on have landed, so the device can start on the first bytes
// of a row before the last has arrived.
//
//   split_part       one part's sub-reads
//   RowGeometry      a row's sub-reads, pieces and their dependencies, from row_geometry()
//   second_stage_start / row_geometry_two_span   a two-stage CPU miss's cut: the first stage's bytes over every root,
//                    then the second's
//   piece_word / publish_piece   the readiness word's format and its publish rule
//   PieceTarget / PiecePublish   where the owner publishes a row's pieces
#pragma once

#include "read_fault.h"

namespace sglang {
namespace expert_stream {

// Splits part `e` into its sub-reads, in file order, into `out` (at most `per_part` entries, per_part <= kSubReads).
// Returns how many.
//
// Each sub-read is len_k = round_up(ceil(length / per_part), kPage) bytes and the last takes what is left, so a
// part tiles exactly, a small part gives fewer than per_part and no sub-read is empty. A zero-length part gives
// none. The part's offset, dest and length are whole pages (the builder's), so every sub-read's are too.
inline int split_part(const Read& e, int per_part, Read* out) {
  if (e.length <= 0) return 0;
  const int64_t len_k = ((e.length + per_part - 1) / per_part + kPage - 1) / kPage * kPage;
  int n = 0;
  for (int64_t at = 0; at < e.length && n < per_part; at += len_k) {
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

// A row's sub-reads and pieces, computed at admission from the row's start.
//
// Sub-read s (its ordinal in the row's file order) is part part[s]'s sub-read k[s]. Piece j covers, in each segment
// i, the run runs[j * segments + i] of the caller's array, and depends on the sub-reads in deps[j] (a bitmask).
struct RowGeometry {
  int subs = 0;
  int prefix = 0;     // row_geometry_two_span: the sub-reads (and so pieces) of the first span; 0: one span
  int64_t start = 0;  // where the row's needed bytes begin in its bounce slot
  Read sub[kPieces] = {};
  int part[kPieces] = {};
  int k[kPieces] = {};
  uint8_t deps[kPieces] = {};
};

// Cuts the sub-reads in `g` (in dest order) into the row's pieces, into g.deps and `runs`, as row_geometry describes.
// Returns false when a piece has bytes that no sub-read reads.
inline bool cut_pieces(const Tables& t, RowGeometry& g, PieceRun* runs) {
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

// Fills `g` and `runs` (kPieces * segments entries) for the row at `row_index` of the tables. Returns false for a
// row that cannot be cut: more reading parts than pieces, or a piece with bytes that no sub-read reads.
//
// Piece j's cuts are the row's sub-read boundaries. In every segment, piece j starts where sub-read j starts, mapped
// into the segment's destination coordinates and rounded down to kPieceAlign there (clamped to the segment), and
// ends where piece j + 1 starts. Piece 0 starts at every segment's first byte and the last sub-read's piece ends at
// every segment's last, so the pieces partition the needed bytes; pieces past the row's sub-read count are empty.
// Segments are sorted by source offset (exl3_expert_format.py) and file order is dest order within a row
// (tables_from), so the cuts are monotone. deps[j] is exactly the set of sub-reads whose file bytes the piece's
// bytes touch.
inline bool row_geometry(const Tables& t, size_t row_index, RowGeometry& g, PieceRun* runs) {
  const size_t parts = static_cast<size_t>(t.parts);
  const size_t base = row_index * parts;
  g = RowGeometry{};
  g.start = t.starts[row_index];
  // The reading parts share the pieces (sub_reads_per_part). A zero-length part (a 0 mirror weight) reads nothing and
  // takes none, so 1:0:1 cuts like a 2-part row.
  int reading = 0;
  for (size_t p = 0; p < parts; ++p)
    reading += t.extents[base + p].length > 0 ? 1 : 0;
  if (reading > kPieces) return false;
  const int per_part = sub_reads_per_part(reading);
  for (size_t p = 0; p < parts; ++p) {
    Read split[kSubReads];
    const int n = split_part(t.extents[base + p], per_part, split);
    if (g.subs + n > kPieces) return false;
    for (int k = 0; k < n; ++k) {
      g.sub[g.subs] = split[k];
      g.part[g.subs] = static_cast<int>(p);
      g.k[g.subs] = k;
      ++g.subs;
    }
  }
  return cut_pieces(t, g, runs);
}

// Where the second stage of a row image begins: the first byte of the first segment whose name has a bit in `second`
// (ExpertRowLayout's kSecondStageMask), when every other segment ends at or before it. 0 when no segment is named or the
// first stage's bytes do not all come first: the image cannot be read in two spans.
inline int64_t second_stage_start(const Tables& t, uint32_t second) {
  int64_t start = INT64_MAX, first_end = 0;
  for (const Segment& s : t.segments) {
    if ((second >> s.name & 1u) != 0) {
      start = std::min(start, s.src);
    } else {
      first_end = std::max(first_end, s.src + s.bytes);
    }
  }
  return start == INT64_MAX || first_end > start ? 0 : start;
}

// A two-stage CPU miss's cut (SGLANG_DSV41_CPU_TWO_STAGE): the row's needed bytes as two spans, the first stage's
// [0, split) and the second's [split, need_end), each divided over the reading parts in the shares the row's parts
// read today, each part's file read at its own offsets (a mirror root holds the whole file). The split is rounded up
// to a page, so the first span holds every first-stage byte, and the inner cuts are pages. Sub-reads are in dest
// order, the first span's first: queued in that order, every root reads its first-span bytes before its second-span
// bytes. A part's first-span sub-read is k = 0 and its second-span one k = 1; a share rounded to nothing is left out.
// g.prefix is the first span's sub-read count, so pieces [0, g.prefix) hold every first-stage byte (cut_pieces maps
// a piece's cut down, never past the split). Falls back to row_geometry's cut when the split is not inside the row or
// the two spans would need more sub-reads than pieces (more than kPieces / 2 reading parts).
inline bool row_geometry_two_span(const Tables& t, size_t row_index, int64_t split, RowGeometry& g, PieceRun* runs) {
  const size_t parts = static_cast<size_t>(t.parts);
  const size_t base = row_index * parts;
  int reading = 0;
  int64_t total = 0;
  for (size_t p = 0; p < parts; ++p) {
    reading += t.extents[base + p].length > 0 ? 1 : 0;
    total += t.extents[base + p].length;
  }
  const int64_t at = (split + kPage - 1) / kPage * kPage;
  if (split <= 0 || at >= t.need_end || 2 * reading > kPieces || total <= 0) return row_geometry(t, row_index, g, runs);
  g = RowGeometry{};
  g.start = t.starts[row_index];
  const int64_t spans[3] = {0, at, t.need_end};
  for (int span = 0; span < 2; ++span) {
    const int64_t lo = spans[span], hi = spans[span + 1];
    int64_t share = 0, from = lo;
    int left = reading;
    for (size_t p = 0; p < parts; ++p) {
      const Read& e = t.extents[base + p];
      if (e.length <= 0) continue;
      share += e.length;
      const int64_t to = --left == 0 ? hi : lo + (hi - lo) * share / total / kPage * kPage;
      if (to <= from) continue;
      g.sub[g.subs] = Read{e.file, e.offset - e.dest + g.start + from, to - from, g.start + from};
      g.part[g.subs] = static_cast<int>(p);
      g.k[g.subs] = span;
      ++g.subs;
      from = to;
    }
    if (span == 0) g.prefix = g.subs;
  }
  return cut_pieces(t, g, runs);
}

// The initial readiness word of a lease row's piece area: generation56 << 8 | bits8, with no bit set. The service
// writes it at reservation and the reader's owner then sets one bit per published piece.
inline uint64_t piece_word(uint64_t generation) {
  return (generation & ((uint64_t{1} << 56) - 1)) << 8;
}

// Sets `bit` in `word` only while the word still carries `generation` and does not have the bit; otherwise returns
// false and leaves the word untouched. A late publish from an older request fails the generation check instead of
// setting a bit under the new one, and a piece published twice fails the bit check.
//
// Release ordering: the caller has already acquired the piece's bytes (the reap's acquire of the completion queue
// tail), so the device that acquires the bit sees them.
inline bool publish_piece(uint64_t* word, uint64_t generation, uint8_t bit) {
  const uint64_t expected = generation & ((uint64_t{1} << 56) - 1);
  uint64_t old = __atomic_load_n(word, __ATOMIC_RELAXED);
  while (true) {
    if ((old >> 8) != expected || (old & bit) != 0) return false;
    if (__atomic_compare_exchange_n(word, &old, old | bit, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) return true;
  }
}

// The most readiness words one row is published to: one per lane, so Wire::kLanes.
constexpr int kPieceTargets = ::sglang::expert_stream::wire::Wire::kLanes;  // one readiness word per lane

// Where the owner publishes a row's pieces: every readiness word naming the row, one per lane.
struct PieceTarget {
  uint64_t* words[kPieceTargets] = {};
  int count = 0;
};
// The publish destinations of one read: the request's generation and, per row, its targets. `rows` is indexed by the
// row's ordinal in the read.
struct PiecePublish {
  uint64_t generation = 0;
  const PieceTarget* rows = nullptr;
};

}  // namespace expert_stream
}  // namespace sglang
