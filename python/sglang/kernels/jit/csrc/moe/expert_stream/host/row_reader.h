// RowReader: io_uring O_DIRECT reads of whole expert rows straight into the pinned slab rows (row images): no bounce,
// no packing.
#pragma once

#include <stdexcept>

#include "reader_core.h"

namespace sglang {
namespace expert_stream {

// Direct mode (row images, Tables::images): there is no bounce and no packing pool. Each read (a part, or with piece
// streaming a sub-read) is ONE O_DIRECT readv whose iovecs are the destination slab rows its image bytes belong in
// (image_iovecs), so the drive writes the caller's unpublished slot directly and nothing is copied. A bounce slot
// index still names the row's pipeline state (rows_, descriptors, banks) but no memory. A row is finished once its
// reads have landed and been vetted; with piece streaming piece j is sub-read j, published as soon as it is vetted
// (publish_landed). The failure rules are the bounce path's, with the unit being the slab bytes themselves: I/O lands
// only in slots the caller has not published, the ring is drained before read() returns, and a failed read leaves
// published whole pieces (never a torn one) and otherwise bytes of unpublished slots the caller releases.
// Trace: with no packing, row_pack_start/end are the clocks of a row's first and last publish (piece streaming)
// or both the clock it was finished (without), pack_workers and pack_split are 0, and useful_bytes still counts its
// segments.
template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build>
class RowReader : public ReaderCore<RowReader<Layout, Reader, Build>, Layout, Reader, Build> {
  using Base = ReaderCore<RowReader<Layout, Reader, Build>, Layout, Reader, Build>;
  friend class ReaderCore<RowReader<Layout, Reader, Build>, Layout, Reader, Build>;
  using Base::c_;
  using Base::direct_;
  using Base::faults_;
  using Base::fds_;
  using Base::finish_row;
  using Base::holding_for_probe;
  using Base::piece_runs_;
  using Base::piece_stream_;
  using Base::publish_collected;
  using Base::rows_;
  using Base::t_;
  using Base::take_ready_row;
  using Base::trace_stamp;
  using typename Base::BounceRow;
  using typename Base::Call;
  using typename Base::ExtentDesc;
  using typename Base::RowState;

 public:
  // Row images read with O_DIRECT are the only reader (plan 2026-09-29-hotpath-zero-overhead D4): shard tables would
  // need the bounce-and-pack copy, and a buffered read copies through the page cache.
  RowReader(Tables tables, bool direct) : Base(std::move(tables), direct) {
    if (!t_.images) {
      throw std::invalid_argument(
          error_prefix<Layout>() + "the reader reads row image tables only (build them with "
          "scripts/dsv41/build_row_images.py); shard tables were refused");
    }
    if (!direct) throw std::invalid_argument(error_prefix<Layout>() + "row images are read with O_DIRECT only");
  }

  // Kept for the stage record's schema-5 fields: a row-image read never packs.
  unsigned pack_workers() const {
    return 0;
  }
  unsigned pack_split() const {
    return 0;
  }

 private:
  static constexpr bool kScatter = true;  // one IORING_OP_READV per read, its iovecs the slab rows (prep_readv)

  // ReaderCore's hooks (see its class comment), the direct path's side, and the row-image machinery.

  // Direct mode publishes each piece as it is vetted (publish_landed): no workers needed.
  void check_piece_stream_support() const {}

  void on_piece_stream_set() {}

  // Direct mode: no bounce; the drive writes the slab rows, so their alignment is checked instead.
  bool open_memory() {
    check_image_alignment();
    return true;
  }

  void open_workers(cpu_set_t /*inherited*/) {}

  // The memory reads land in, registered with the ring: one region per named slab, with its row size. In a fixed read
  // mode every slab is checked against them first (check_slab_regions).
  std::vector<RegisteredRegion> registered_regions() const {
    if (this->fixed_reads()) check_slab_regions();
    return t_.buffer_regions;
  }

  // Fixed read modes: each slab must start on a row boundary of one registered region whose row size is the slab's
  // own, with room for a row, as _table_buffer_regions builds them from tables.keepalive. A slab missing from
  // keepalive, or registered with another row size, would otherwise fail its first read mid-serve on the service
  // thread ("lies in no registered buffer"); refuse at open instead, naming the slab. The slot range is the caller's
  // (read_rows_once and the tier check slots against tables.capacity), so rows past the region still fail at read
  // time. A null slab (a layer with no rows) is skipped.
  void check_slab_regions() const {
    for (size_t layer = 0; layer < t_.slabs.size(); ++layer) {
      for (size_t name = 0; name < t_.slabs[layer].size(); ++name) {
        const auto base = reinterpret_cast<uintptr_t>(t_.slabs[layer][name]);
        const auto row = static_cast<uint64_t>(t_.row_bytes[name]);
        if (base == 0 || row == 0) continue;
        const std::string slab = "slab " + std::string(Layout::kNames[name]) + " of layer " + std::to_string(layer);
        const RegisteredRegion* holder = nullptr;
        for (const RegisteredRegion& r : t_.buffer_regions) {
          const auto start = reinterpret_cast<uintptr_t>(r.base);
          if (base >= start && base - start < r.bytes) {
            holder = &r;
            break;
          }
        }
        if (holder == nullptr) {
          throw std::runtime_error(
              error_prefix<Layout>() + "fixed reads: " + slab +
              " lies in no registered buffer region (is its tensor missing from tables.keepalive?)");
        }
        const uint64_t offset = base - reinterpret_cast<uintptr_t>(holder->base);
        if (holder->row_bytes != row || offset % row != 0 || offset + row > holder->bytes) {
          throw std::runtime_error(
              error_prefix<Layout>() + "fixed reads: " + slab + " (row_bytes " + std::to_string(row) +
              ") lies in a registered buffer region with row_bytes " + std::to_string(holder->row_bytes) +
              " at offset " + std::to_string(offset) + " of its " + std::to_string(holder->bytes) +
              " bytes; each slab needs a region of its own row size, starting on one of its rows");
        }
      }
    }
  }

  size_t max_iovecs() const {
    return t_.segments.size();
  }

  // Where descriptor `d`'s remaining bytes land, as iovecs in `out` (max_iovecs of them). Returns how many.
  unsigned destination(const ExtentDesc& d, iovec* out) const {
    return image_iovecs(size_t(d.slot), d.read->dest + d.done, d.read->dest + d.read->length, out);
  }

  // Direct mode's turn of work: publish what has landed (publish_landed).
  bool advance() {
    return publish_landed();
  }

  // Direct mode's "packing": nothing to copy, the drive already wrote the slab rows. Without piece streaming, finish
  // every row whose reads have landed and whose bytes take_ready_row vetted. With it, publish every vetted piece at
  // once (piece j is exactly sub-read j's bytes, vetted when it landed, and the reap's acquire of the CQ tail orders
  // the drive's writes before the release that sets the bit), and finish a row once all its pieces are published and
  // its sub-reads retired. One clock read per turn stamps what it published (pack stamps = publish time); a piece with
  // no bytes (past the row's sub-reads) is published at admission like the bounce path's, and stamps nothing, so a
  // row's stamps are those of its landed bytes. The pack_delay_ns fault delays each publish (a slow publisher).
  bool publish_landed() {
    Call& c = c_;
    bool any = false;
    int64_t now = -1;
    const auto clock = [&] {
      if (now < 0) now = trace_stamp();
      return now;
    };
    const auto delay = [&] {  // the pack_delay_ns fault: InstrBuild only
      if constexpr (Build::kFaults) {
        if (faults_.fault.pack_delay_ns <= 0) return;
        std::this_thread::sleep_for(std::chrono::nanoseconds(faults_.fault.pack_delay_ns));
        now = -1;
      }
    };
    if (!piece_stream_) {
      while (true) {
        const size_t best = take_ready_row();
        if (best == static_cast<size_t>(kBounceSlots)) return any;
        delay();
        finish_row(best, clock(), clock());
        any = true;
        if constexpr (Build::kFaults) {
          // The slow publisher (pack_delay_ns) finishes one row per turn, so the loop reaps between rows as a real
          // publisher's would: without it, rows 1..n that landed before row 0 are all finished (n delays) before row 0
          // is even reaped, and a claim-order prefix stalls behind a row that landed long ago. InstrBuild only;
          // ProdBuild's finish has no delay and takes every ready row in one pass.
          if (faults_.fault.pack_delay_ns > 0) return any;
        }
      }
    }
    const size_t segments = t_.segments.size();
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots) && !c.failed; ++s) {
      BounceRow& r = rows_[s];
      if (r.state == RowState::Free) continue;
      const uint8_t todo = r.vetted & static_cast<uint8_t>(~r.dispatched);
      for (int j = 0; j < kPieces && !c.failed && todo != 0; ++j) {
        const uint8_t bit = static_cast<uint8_t>(1u << j);
        if ((todo & bit) == 0) continue;
        if (j >= 1 && holding_for_probe()) continue;
        const PieceRun* runs = &piece_runs_[(s * kPieces + static_cast<size_t>(j)) * segments];
        bool bytes = false;
        for (size_t i = 0; i < segments && !bytes; ++i)
          bytes = runs[i].lo < runs[i].hi;
        r.dispatched |= bit;
        if (bytes) {
          delay();
          r.pack_first = std::min(r.pack_first, clock());
          r.pack_last = std::max(r.pack_last, clock());
        }
        publish_collected(s, j);
        any = true;
      }
      if (!c.failed && r.state == RowState::Ready && r.published == kAllPieces) {
        finish_row(s, r.pack_first == INT64_MAX ? 0 : r.pack_first, r.pack_last);
      }
    }
    return any;
  }

  // Direct mode: the iovecs that land image bytes [from, to) of the row in `slot` in its destination slab rows, one
  // per segment the range meets (the segments tile the image in source order: check_image_tables). Returns how many.
  unsigned image_iovecs(size_t slot, int64_t from, int64_t to, iovec* out) const {
    const Call& c = c_;
    const int64_t dest = c.slots[rows_[slot].ordinal];
    unsigned count = 0;
    for (const Segment& s : t_.segments) {
      const int64_t lo = std::max(from, s.src), hi = std::min(to, s.src + s.bytes);
      if (lo >= hi) continue;
      out[count++] = iovec{
          t_.slabs[c.layer][s.name] + dest * t_.row_bytes[s.name] + s.dst + (lo - s.src), static_cast<size_t>(hi - lo)};
    }
    return count;
  }

  // The poison fault: fill the destination slab rows the row in `slot` is read into, so a byte published without
  // having been read shows. InstrBuild only.
  void poison_slot(size_t slot, uint8_t fill)
    requires(Build::kFaults)
  {
    const Call& c = c_;
    const int64_t dest = c.slots[rows_[slot].ordinal];
    for (size_t name = 0; name < t_.slabs[c.layer].size(); ++name) {
      std::memset(t_.slabs[c.layer][name] + dest * t_.row_bytes[name], fill, static_cast<size_t>(t_.row_bytes[name]));
    }
  }

  // Direct mode, at open: O_DIRECT refuses (EINVAL) a file offset or segment length off the logical block, so a
  // table that would ask for one is refused here instead of failing reads at run time. Every slab row base, every
  // segment's bounds (the iovec cuts) and every reading extent's offset, length and image position must be
  // kImageAlign multiples; the sub-read cuts inside a part are pages (split_part), so they follow.
  // With O_DIRECT, each file's own alignment (statx STATX_DIOALIGN, where the kernel reports it) is checked too, so a
  // drive with larger logical blocks than the 512 B the images are built for is refused at open, naming the file.
  void check_image_alignment() const {
    const auto off = [](int64_t v) { return v % kImageAlign != 0; };
#ifdef STATX_DIOALIGN
    for (size_t f = 0; f < fds_.size() && direct_; ++f) {
      struct statx stx{};
      if (statx(fds_[f], "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) != 0 || (stx.stx_mask & STATX_DIOALIGN) == 0)
        continue;
      if (stx.stx_dio_offset_align == 0 || kImageAlign % stx.stx_dio_offset_align != 0 || stx.stx_dio_mem_align == 0 ||
          kImageAlign % stx.stx_dio_mem_align != 0) {
        throw std::runtime_error(
            error_prefix<Layout>() + t_.paths[f] + " needs O_DIRECT alignment of " +
            std::to_string(stx.stx_dio_offset_align) + " B (offsets) and " + std::to_string(stx.stx_dio_mem_align) +
            " B (memory); row images are built for 512 B");
      }
    }
#endif
    for (size_t row = 0; row < t_.slabs.size(); ++row) {
      for (size_t name = 0; name < t_.slabs[row].size(); ++name) {
        if (reinterpret_cast<uintptr_t>(t_.slabs[row][name]) % kImageAlign != 0 || off(t_.row_bytes[name])) {
          throw std::runtime_error(error_prefix<Layout>() + "row images need every slab row 512 B aligned");
        }
      }
    }
    for (const Segment& s : t_.segments) {
      if (off(s.src) || off(s.dst) || off(s.bytes)) {
        throw std::runtime_error(error_prefix<Layout>() + "row images need every segment 512 B aligned");
      }
    }
    for (const Read& e : t_.extents) {
      if (e.length > 0 && (off(e.offset) || off(e.length) || off(e.dest))) {
        throw std::runtime_error(
            error_prefix<Layout>() + "row images need every extent's offset and length 512 B aligned");
      }
    }
  }

  // Direct mode: no piece is ever held by a job (publish_landed publishes on dispatch), so the only collecting left
  // is, with piece streaming, finishing each row whose pieces are all published once its last sub-read retires.
  void collect() {
    if (!piece_stream_) return;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      BounceRow& r = rows_[s];
      if (r.state == RowState::Ready && r.published == kAllPieces) {
        finish_row(s, r.pack_first == INT64_MAX ? 0 : r.pack_first, r.pack_last);
      }
    }
  }

  // Direct mode holds no copies (c.packing is always 0), so there is nothing to wait for.
  void quiesce() {}

  // Direct mode: the bytes are the slot's own now, and the caller publishes them.
  void after_finish(size_t /*slot*/) {}
};

}  // namespace expert_stream
}  // namespace sglang
