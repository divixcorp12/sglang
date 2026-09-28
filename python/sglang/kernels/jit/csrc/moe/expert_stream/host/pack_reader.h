// PackReader: io_uring superset reads of whole expert rows into page-aligned bounce banks, then per-name splits into
// pinned slabs.
#pragma once

#include "pack_pool.h"
#include "reader_core.h"

namespace sglang {
namespace expert_stream {

// The bounce-and-pack path (Tables::images false): every read lands in a page-aligned bounce slot, and each row, or
// with piece streaming each piece, is copied into the caller's slab rows on the owner or on packing workers
// (set_pack). ReaderCore runs the pipeline; this class owns the bounce, the packing pool and its jobs.
template <ExpertRowLayout Layout, AsyncFileReader Reader>
class PackReader : public ReaderCore<PackReader<Layout, Reader>, Layout, Reader> {
  using Base = ReaderCore<PackReader<Layout, Reader>, Layout, Reader>;
  friend class ReaderCore<PackReader<Layout, Reader>, Layout, Reader>;
  using Base::c_;
  using Base::close_io;
  using Base::fault_;
  using Base::finish_row;
  using Base::holding_for_probe;
  using Base::kPoisonFill;
  using Base::piece_runs_;
  using Base::piece_stream_;
  using Base::publish_collected;
  using Base::rows_;
  using Base::t_;
  using Base::take_ready_row;
  using typename Base::BounceRow;
  using typename Base::Call;
  using typename Base::ExtentDesc;
  using typename Base::RowState;

 public:
  PackReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0)
      : Base(std::move(tables), direct) {
    if (t_.images) throw std::runtime_error(error_prefix<Layout>() + "row image tables are read by RowReader");
    set_pack(pack_workers, pack_split);
  }

  ~PackReader() {
    pool_.reset();  // joins the workers before the bounce they read from is freed
    close_io();     // unregisters the bounce before it is freed
    std::free(bounce_);
  }

  // Pack on `workers` copy threads instead of the owner (0: on the owner, the default), each row in
  // `split` byte-range chunks (0: one per worker); with piece streaming each piece, about an eighth of a
  // row, is cut that way instead. Takes effect at open().
  void set_pack(int64_t workers, int64_t split) {
    pack_workers_ = static_cast<unsigned>(std::max<int64_t>(0, workers));
    pack_split_ = split > 0 ? static_cast<unsigned>(split) : pack_workers_;
  }

  unsigned pack_workers() const {
    return pack_workers_;
  }

  unsigned pack_split() const {
    return pack_split_;
  }

  std::vector<int> packing_cpus() const {
    return pool_ ? pool_->cpus() : std::vector<int>{};
  }

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

 private:
  static constexpr bool kScatter = false;  // one IORING_OP_READ per read into the bounce (prep_read)

  // ReaderCore's hooks (see its class comment), the bounce path's side, and the bounce and packing machinery.

  uint8_t* bounce_slot(size_t slot) const {
    return bounce_ + slot * static_cast<size_t>(t_.slot_bytes);
  }

  // Piece streaming publishes from the owner once a job is done, so the bounce path needs packing workers.
  void check_piece_stream_support() const {
    if (pack_workers_ == 0) {
      throw std::runtime_error(error_prefix<Layout>() + "piece streaming needs packing workers");
    }
  }

  void on_piece_stream_set() {
    if (pool_) size_jobs();
  }

  // The bounce: kBounceSlots page-aligned row supersets.
  bool open_memory() {
    if (posix_memalign(reinterpret_cast<void**>(&bounce_), kPage, static_cast<size_t>(kBounceSlots * t_.slot_bytes)) !=
        0) {
      bounce_ = nullptr;
      return false;
    }
    return true;
  }

  void open_workers(cpu_set_t inherited) {
    if (piece_stream_) check_piece_stream_support();
    if (pack_workers_ > 0) {
      // Every buffer the workers use is sized here too: a job and a run list per bounce slot.
      runs_.assign(static_cast<size_t>(kBounceSlots) * t_.segments.size(), CopyRun{});
      pool_ = std::make_unique<PackPool>(
          pack_workers_,
          inherited,
          static_cast<size_t>(kBounceSlots),
          error_prefix<Layout>(),
          std::string(Layout::kName) + "-pack");
      if (piece_stream_) size_jobs();
    }
  }

  // The memory reads land in, registered with the ring: one region per bounce slot, or the slab rows.
  std::vector<iovec> registered_regions() const {
    std::vector<iovec> regions;
    for (size_t slot = 0; slot < static_cast<size_t>(kBounceSlots); ++slot)
      regions.push_back({bounce_slot(slot), static_cast<size_t>(t_.slot_bytes)});
    return regions;
  }

  size_t max_iovecs() const {
    return 1;
  }

  // Where descriptor `d`'s remaining bytes land, as iovecs in `out` (max_iovecs of them). Returns how many.
  unsigned destination(const ExtentDesc& d, iovec* out) const {
    out[0] = {bounce_slot(d.slot) + d.read->dest + d.done, size_t(d.read->length - d.done)};
    return 1;
  }

  // Piece streaming packs a piece per job: a job and a run list per (bounce slot, piece), and a packing queue that
  // holds every one of them. With the flag off, a job and a run list per bounce slot, as open() sizes them.
  void size_jobs() {
    const size_t jobs = static_cast<size_t>(kBounceSlots) * (piece_stream_ ? kPieces : 1);
    runs_.assign(jobs * t_.segments.size(), CopyRun{});
    pool_->set_capacity(jobs);
  }

  // Pack ONE complete row inline, the earliest in request order among those ready. One row per loop turn
  // keeps packing bounded: the loop refills and reaps between rows. With a packing pool, every ready row
  // is handed to the workers instead, and the loop finishes each one when its copy is done.
  bool advance() {
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
          t_.slabs[c.layer][segment.name] + slot * t_.row_bytes[segment.name] + segment.dst,
          base + segment.src,
          static_cast<size_t>(segment.bytes));
    }
    finish_row(best, start, stamp(c.trace));
    return true;
  }

  // The poison fault: fill the memory the row in `slot` is read into, the bounce slot or (direct mode) the
  // destination slab rows, so a byte published without having been read shows.
  void poison_slot(size_t slot, uint8_t fill) {
    std::memset(bounce_slot(slot), fill, static_cast<size_t>(t_.slot_bytes));
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
      if (!jobs_[best].done())
        throw std::runtime_error(error_prefix<Layout>() + "a packing job was re-armed while a worker still holds it");
      CopyRun* runs = &runs_[best * t_.segments.size()];
      for (size_t i = 0; i < t_.segments.size(); ++i) {
        const Segment& segment = t_.segments[i];
        runs[i] = CopyRun{
            t_.slabs[c.layer][segment.name] + slot * t_.row_bytes[segment.name] + segment.dst,
            base + segment.src,
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
              base + segment.src + piece[i].lo,
              piece[i].hi - piece[i].lo};
          bytes += runs[i].bytes;
        }
        r.dispatched |= bit;
        if (bytes == 0) {
          publish_collected(s, j);
          continue;
        }
        PackJob& job = jobs_[job_index];
        if (!job.done())
          throw std::runtime_error(error_prefix<Layout>() + "a packing job was re-armed while a worker still holds it");
        job.arm(runs, segments, pack_split_, fault_.pack_delay_ns, c.trace ? &worker_stamp : nullptr, c.trace);
        pool_->post(&job);  // throws before queueing
        ++c.packing;
        any = true;
      }
    }
    return any;
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

  // Finish every row whose copy the workers have completed: the bank's packing reference is released
  // here, on the owner, and only after its job reads done.
  void collect() {
    Call& c = c_;
    if (piece_stream_) return collect_pieces();
    if (c.packing == 0) return;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (rows_[s].state != RowState::Packing || !jobs_[s].done()) continue;
      finish_row(
          s, jobs_[s].first_start.load(std::memory_order_relaxed), jobs_[s].last_end.load(std::memory_order_relaxed));
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
          while (!jobs_[s * kPieces + static_cast<size_t>(j)].done())
            _mm_pause();
        }
      }
    }
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (rows_[s].state != RowState::Packing) continue;
      while (!jobs_[s].done())
        _mm_pause();
    }
    collect();
  }

  // The poison fault re-fills a finished row's bounce slot, so a later row that reuses it without reading shows.
  void after_finish(size_t slot) {
    if (fault_.poison) poison_slot(slot, kPoisonFill ^ 0xFF);
  }

  uint8_t* bounce_ = nullptr;
  // Packing workers (set_pack; none by default): a job and a run list per bounce slot, sized at open().
  unsigned pack_workers_ = 0;
  unsigned pack_split_ = 0;
  // io_ lives in ReaderCore, so it is destroyed after every member here. ~PackReader therefore joins the
  // workers (pool_.reset()) and closes io_ explicitly (close_io()) before std::free(bounce_): the ring
  // unregisters the bounce while it is still allocated. read() always drains the ring (quiesce()) before
  // returning, so nothing is ever in flight when this reader is destroyed.
  std::unique_ptr<PackPool> pool_;
  // Indexed by bounce slot with the flag off, by (slot, piece) with piece streaming (size_jobs).
  PackJob jobs_[kBounceSlots * kPieces];
  std::vector<CopyRun> runs_;
};

}  // namespace expert_stream
}  // namespace sglang
