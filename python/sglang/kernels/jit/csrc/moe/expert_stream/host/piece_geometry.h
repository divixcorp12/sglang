// Piece streaming geometry: sub-read splitting and the piece publish record.
#pragma once

#include "read_fault.h"

namespace sglang {
namespace expert_stream {

// Piece streaming, sub-reads (plan §4.1): part `e` as its sub-reads, in file order, into `out` (at most `per_part`
// entries, per_part <= kSubReads); returns how many. Each is len_k = round_up(ceil(length / per_part), kPage) bytes and
// the last takes what is left, so a part tiles exactly, a small part gives fewer than per_part and no sub-read is
// empty. A zero-length part gives none. The part's offset, dest and length are whole pages (the builder's), so every
// sub-read's are too.
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
// row that cannot be cut: more reading parts than pieces, or a piece with bytes that no sub-read reads.
inline bool row_geometry(const Tables& t, size_t row_index, RowGeometry& g, PieceRun* runs) {
  const size_t parts = static_cast<size_t>(t.parts);
  const size_t base = row_index * parts;
  g = RowGeometry{};
  g.start = t.starts[row_index];
  // The row's reading parts share the pieces (sub_reads_per_part); a zero-length part (a 0 mirror weight) reads nothing
  // and takes none, so 1:0:1 cuts like a 2-part row.
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
};

}  // namespace expert_stream
}  // namespace sglang
