// RowReader: io_uring superset reads of whole expert rows into page-aligned bounce banks, then per-name splits into
// pinned slabs.
#pragma once

#include "../row_layout.h"
#include "file_reader.h"
#include "pack_pool.h"
#include "piece_geometry.h"
#include "uring_options.h"

namespace sglang {
namespace expert_stream {

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
//
// Direct mode (row images, Tables::images): there is no bounce and no packing pool. Each read (a part, or with piece
// streaming a sub-read) is ONE O_DIRECT readv whose iovecs are the destination slab rows its image bytes belong in
// (image_iovecs), so the drive writes the caller's unpublished slot directly and nothing is copied. A bounce slot
// index still names the row's pipeline state (rows_, descriptors, banks) but no memory. A row is finished once its
// reads have landed and been vetted; with piece streaming piece j is sub-read j, published as soon as it is vetted
// (publish_landed). The failure rules are the bounce path's, with the unit being the slab bytes themselves: I/O lands
// only in slots the caller has not published, the ring is drained before read() returns, and a failed read leaves
// published whole pieces (never a torn one) and otherwise bytes of unpublished slots the caller releases.
// Trace: with no packing, row_pack_start/end are the clocks of a row's first and last publish (piece streaming)
// or both the clock it was finished (without), pack_workers is 0, and useful_bytes still counts its segments.
template <ExpertRowLayout Layout, AsyncFileReader Reader>
class RowReader {
 public:
  using LayoutType = Layout;  // named so it cannot shadow the template parameter

  RowReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0)
      : t_(std::move(tables)), direct_(direct) {
    configured_queue_depth_ = UringOptions::from_env().queue_depth;
    set_pack(pack_workers, pack_split);
  }
  RowReader(const RowReader&) = delete;  // owns fds, the ring and the bounce
  RowReader& operator=(const RowReader&) = delete;

  ~RowReader() {
    pool_.reset();  // joins the workers before the bounce they read from is freed
    // Registered regions must be released while their allocations and files still exist.
    if constexpr (requires(Reader& reader) { reader.close(); }) io_.close();
    for (int fd : fds_)
      ::close(fd);
    std::free(bounce_);
    // Undo the owner-pin scaffold's affinity change: the pin targets the calling thread, which a caller
    // (e.g. the benchmark) may reuse across many readers, so a later open() must see the original mask,
    // not the single core this reader pinned itself to.
    if (owner_pinned_) pthread_setaffinity_np(pthread_self(), sizeof(unpinned_affinity_), &unpinned_affinity_);
  }

  const Tables& tables() const {
    return t_;
  }

  // Pack on `workers` copy threads instead of the owner (0: on the owner, the default), each row in
  // `split` byte-range chunks (0: one per worker); with piece streaming each piece, about an eighth of a
  // row, is cut that way instead. Takes effect at open().
  void set_pack(int64_t workers, int64_t split) {
    // Direct mode copies nothing, so it never starts a pool: the setting is ignored (and traced as 0).
    pack_workers_ = t_.images ? 0u : static_cast<unsigned>(std::max<int64_t>(0, workers));
    pack_split_ = split > 0 ? static_cast<unsigned>(split) : pack_workers_;
  }
  unsigned pack_workers() const {
    return t_.images ? row_pack_workers() : pack_pack_workers();
  }
  unsigned pack_split() const {
    return t_.images ? row_pack_split() : pack_pack_split();
  }
  std::vector<int> packing_cpus() const {
    return t_.images ? row_packing_cpus() : pack_packing_cpus();
  }

  // Piece streaming (sub_reads_per_part(reading) sub-reads per reading part, per-piece vetting, packing and
  // publishing); off by default. Before open(), or on an idle reader after it (the tier sets it before its service
  // thread starts), since it resizes the descriptor arrays and the packing queue. Refused without packing workers
  // (the inline path has no piece publisher), with more mirror parts than pieces (a reading part needs a piece of
  // its own), or when a slab row base is not kPieceAlign-aligned (a piece's cuts are aligned in the row).
  void set_piece_stream(bool on) {
    if (on) {
      check_piece_stream_support();
      if (t_.parts > kPieces) {
        throw std::runtime_error(
            error_prefix<Layout>() + "piece streaming reads at most " + std::to_string(kPieces) +
            " mirror parts (a reading part needs a piece of its own), not " + std::to_string(t_.parts));
      }
      for (size_t row = 0; row < t_.slabs.size(); ++row) {
        for (size_t name = 0; name < t_.slabs[row].size(); ++name) {
          if (reinterpret_cast<uintptr_t>(t_.slabs[row][name]) % kPieceAlign != 0 ||
              t_.row_bytes[name] % kPieceAlign != 0) {
            throw std::runtime_error(
                error_prefix<Layout>() + "piece streaming needs every slab row base 128 B aligned");
          }
        }
      }
    }
    piece_stream_ = on;
    subs_ = on ? kSubReads : 1;
    if (io_.ready() && !size_extents())
      throw std::runtime_error(error_prefix<Layout>() + "too many descriptors for piece streaming");
    on_piece_stream_set();
  }
  bool piece_stream() const {
    return piece_stream_;
  }
  // Pieces a readiness word refused to publish, over the reader's life (each also failed its read).
  int64_t publish_refused() const {
    return publish_refused_;
  }
  // Test only (U10): the descriptor count and the ring credit this reader runs with.
  size_t descriptors() const {
    return descs_.size();
  }
  unsigned credit() const {
    return queue_depth();
  }
  // Test only (U10): every SQE prepared is appended here, while set (null: nothing recorded, one branch per SQE).
  struct SqeRecord {
    int64_t file, offset, length, bounce;  // bounce: byte offset of the destination from the bounce's start
  };
  void set_sqe_log(std::vector<SqeRecord>* log) {
    sqe_log_ = log;
  }

  // Test-only scaffold (PACK_WORKERS.md owner-pinning measurement): pin the owner thread to `core`
  // at open() and build the packing pool's mask as the inherited set minus that core, so the owner and
  // the workers never share a core. -1 (the default) leaves open() byte-for-byte what it is today: no
  // pin, and the pool's mask is exactly the creating thread's inherited affinity.
  void set_owner_core(int64_t core) {
    owner_core_ = core;
  }
  // Copies a worker still holds. read() leaves none: this is what a test checks after it returns.
  int64_t unfinished_jobs() const {
    return t_.images ? row_unfinished_jobs() : pack_unfinished_jobs();
  }

  void set_fault(const ReadFault& fault) {
    fault_ = fault;
    part_fired_ = false;
    retired_ = 0;
    stale_armed_ = false;
    stale_waiting_ = false;
    if (fault.generation_start != 0) generation_ = static_cast<uint32_t>(fault.generation_start);
    if constexpr (requires { io_.set_submit_fault(SubmitFault{}); }) {
      io_.set_submit_fault(
          SubmitFault{fault.submit_error, fault.submit_call, fault.submit_first, fault.submit_short_call});
    } else if (
        fault.submit_error != 0 || fault.submit_call != 0 || fault.submit_first || fault.submit_short_call != 0) {
      // No test reaches this today: the one production instantiation (exl3_ram_miss_host.cpp) pairs RowReader
      // with FaultyReader<UringReader>, which has set_submit_fault, so the `if constexpr` branch above always
      // fires there. This is the fallback for a Reader that cannot inject submit faults at all.
      throw std::runtime_error(error_prefix<Layout>() + "this reader cannot inject submit faults");
    }
  }

  // Completions reaped over the reader's life (tests: a zero-length extent must add none).
  int64_t cqes() const {
    return cqes_;
  }
  // Completions that named no live descriptor, and generation counter wraps (tests).
  int64_t stale_cqes() const {
    return stale_cqes_;
  }
  int64_t generation_wraps() const {
    return generation_wraps_;
  }

  bool open() {
    for (const auto& path : t_.paths) {
      const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct_ ? O_DIRECT : 0));
      if (fd < 0) {
        std::fprintf(
            stderr, "ERROR %sopen %s: %s\n", error_prefix<Layout>().c_str(), path.c_str(), std::strerror(errno));
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
            error_prefix<Layout>() + path + " has size " + std::to_string(st.st_size) + " bytes but its source " +
            t_.source_paths[file] + " has size " + std::to_string(t_.file_sizes[file]) +
            " bytes; the copy is incomplete or stale");
      }
      const int64_t dev = statted ? static_cast<int64_t>(st.st_dev) : -1;
      size_t drive = 0;
      while (drive < devs_.size() && devs_[drive] != dev)
        ++drive;
      if (drive == devs_.size()) devs_.push_back(dev);
      file_drive_.push_back(static_cast<uint8_t>(std::min<size_t>(drive, kMaxDrives - 1)));
    }
    for (size_t drive = 0; drive < devs_.size(); ++drive) {
      const size_t slot = std::min<size_t>(drive, kMaxDrives - 1);
      drive_dev_[slot] = drive < kMaxDrives ? devs_[drive] : -1;
    }
    if (!open_memory()) return false;
    if (!size_extents()) return false;
    if (!io_.init(queue_depth())) return false;
    if constexpr (requires(Reader& reader, const std::vector<int>& files, const std::vector<iovec>& buffers) {
                    reader.configure_resources(files, buffers, true);
                  }) {
      io_.configure_resources(fds_, registered_regions(), direct_);
    }
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
        throw std::runtime_error(error_prefix<Layout>() + "could not pin the owner thread to its core");
      }
      owner_pinned_ = true;
    }
    open_workers(inherited);
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
    if (!io_.ready()) return 0;
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
      trace->pack_workers = pack_workers();
      trace->pack_split = pack_split();
      trace->piece_stream = piece_stream_ ? 1 : 0;
    }
    reset_pipeline();
    held_.clear();
    // Whatever way this call ends, no packing worker may still be copying when it does: the caller
    // releases the slots on return and the next read reuses the bounce. Runs on exceptions too.
    // An exception (one of the accounting guards) leaves reads in flight: drain them too before unwinding past the
    // caller, which releases the slots. In direct mode those reads write the slab rows themselves.
    struct Quiesce {
      RowReader* reader;
      int exceptions;
      ~Quiesce() {
        reader->quiesce();
        if (std::uncaught_exceptions() > exceptions) reader->drain(reader->c_.pending);
      }
    } quiesce_on_exit{this, std::uncaught_exceptions()};
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
      collect();  // before admit: a bank whose last copy just finished is free for the next batch
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
      // (The hold_until_probe_ms fault keeps direct-mode pieces vetted but unpublished with nothing to wait on.)
      if (!advance() && (c.packing > 0 || c.hold_until != 0)) _mm_pause();
    }
    // Nothing in flight and nothing to pack, yet a row was left unread or unpacked, or a batch was
    // neither admitted nor abandoned: the loop's own bookkeeping is wrong. Fail instead of returning a
    // row that was never read.
    if (!c.failed) {
      bool clean = c.reading_rows == 0 && c.queue_count == 0;
      for (int b = 0; b < kBanks; ++b)
        clean = clean && rows_busy_[b] == 0 && bank_live_[b] == 0;
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
    size_t ordinal = 0;  // the row's index in the request
    unsigned extents_left = 0;
    // Coverage, checked before the row is packed. `needed` is the last byte of the slot the segments
    // can read; `filled` is what the drives actually delivered into it. See take_ready_row().
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

  using Completion = ReadCompletion;

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
    int64_t events = 0;      // piece streaming: sub-read landings and piece vettings so far (the trace's sequence)
    size_t published = 0;    // piece streaming: pieces published so far (the last_publish_delay_ns fault)
    int64_t hold_until = 0;  // piece streaming: when the hold_until_probe_ms fault gives up (0: no hold)
  };

  // How many reads may be outstanding at once. Credit-based preparation in refill() means
  // this bounds concurrency, not batch size: a batch larger than the ring waits for credit
  // rather than overrunning it. Scaled by parts so splitting a row across roots does not
  // halve the number of rows in flight.
  unsigned queue_depth() const {
    return configured_queue_depth_ != 0 ? configured_queue_depth_ : kQueueDepth * static_cast<unsigned>(t_.parts);
  }

  uint32_t next_generation() {
    if (++generation_ == 0) {  // 0 means retired: skip it when the counter wraps
      ++generation_;
      ++generation_wraps_;
    }
    return generation_;
  }

  uint8_t* bounce_slot(size_t slot) const {
    return bounce_ + slot * static_cast<size_t>(t_.slot_bytes);
  }

  // Path hooks. Every place the bounce path (pack_*) and the direct path (row_*, Tables::images) differ is one of
  // these; the shared pipeline calls only the dispatchers below.
  void check_piece_stream_support() const {
    t_.images ? row_check_piece_stream_support() : pack_check_piece_stream_support();
  }
  void on_piece_stream_set() {
    t_.images ? row_on_piece_stream_set() : pack_on_piece_stream_set();
  }
  bool open_memory() {
    return t_.images ? row_open_memory() : pack_open_memory();
  }
  void open_workers(cpu_set_t inherited) {
    t_.images ? row_open_workers(inherited) : pack_open_workers(inherited);
  }
  std::vector<iovec> registered_regions() const {
    return t_.images ? row_registered_regions() : pack_registered_regions();
  }
  size_t max_iovecs() const {
    return t_.images ? row_max_iovecs() : pack_max_iovecs();
  }
  unsigned destination(const ExtentDesc& d, iovec* out) const {
    return t_.images ? row_destination(d, out) : pack_destination(d, out);
  }
  bool advance() {
    return t_.images ? row_advance() : pack_advance();
  }
  void collect() {
    t_.images ? row_collect() : pack_collect();
  }
  void quiesce() {
    t_.images ? row_quiesce() : pack_quiesce();
  }
  void poison_slot(size_t slot, uint8_t fill) {
    t_.images ? row_poison_slot(slot, fill) : pack_poison_slot(slot, fill);
  }
  void after_finish(size_t slot) {
    t_.images ? row_after_finish(slot) : pack_after_finish(slot);
  }

  // Piece streaming publishes from the owner once a job is done, so the bounce path needs packing workers.
  void pack_check_piece_stream_support() const {
    if (pack_workers_ == 0) {
      throw std::runtime_error(error_prefix<Layout>() + "piece streaming needs packing workers");
    }
  }
  // Direct mode publishes each piece as it is vetted (publish_landed): no workers needed.
  void row_check_piece_stream_support() const {}

  void pack_on_piece_stream_set() {
    if (pool_) size_jobs();
  }
  void row_on_piece_stream_set() {}

  // The bounce: kBounceSlots page-aligned row supersets.
  bool pack_open_memory() {
    if (posix_memalign(reinterpret_cast<void**>(&bounce_), kPage, static_cast<size_t>(kBounceSlots * t_.slot_bytes)) !=
        0) {
      bounce_ = nullptr;
      return false;
    }
    return true;
  }
  // Direct mode: no bounce; the drive writes the slab rows, so their alignment is checked instead.
  bool row_open_memory() {
    check_image_alignment();
    return true;
  }

  void pack_open_workers(cpu_set_t inherited) {
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
  void row_open_workers(cpu_set_t /*inherited*/) {}

  // The memory reads land in, registered with the ring: one region per bounce slot, or the slab rows.
  std::vector<iovec> pack_registered_regions() const {
    std::vector<iovec> regions;
    for (size_t slot = 0; slot < static_cast<size_t>(kBounceSlots); ++slot)
      regions.push_back({bounce_slot(slot), static_cast<size_t>(t_.slot_bytes)});
    return regions;
  }
  std::vector<iovec> row_registered_regions() const {
    return t_.buffer_regions;
  }

  size_t pack_max_iovecs() const {
    return 1;
  }
  size_t row_max_iovecs() const {
    return t_.segments.size();
  }

  // Where descriptor `d`'s remaining bytes land, as iovecs in `out` (max_iovecs of them). Returns how many.
  unsigned pack_destination(const ExtentDesc& d, iovec* out) const {
    out[0] = {bounce_slot(d.slot) + d.read->dest + d.done, size_t(d.read->length - d.done)};
    return 1;
  }
  unsigned row_destination(const ExtentDesc& d, iovec* out) const {
    return image_iovecs(size_t(d.slot), d.read->dest + d.done, d.read->dest + d.read->length, out);
  }

  int64_t pack_unfinished_jobs() const {
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
  int64_t row_unfinished_jobs() const {
    return 0;
  }

  unsigned pack_pack_workers() const {
    return pack_workers_;
  }
  unsigned row_pack_workers() const {
    return 0;
  }
  // Direct mode keeps the split set_pack stored (traced as given), though nothing is cut by it.
  unsigned pack_pack_split() const {
    return pack_split_;
  }
  unsigned row_pack_split() const {
    return pack_split_;
  }

  std::vector<int> pack_packing_cpus() const {
    return pool_ ? pool_->cpus() : std::vector<int>{};
  }
  std::vector<int> row_packing_cpus() const {
    return {};
  }

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
    // Each descriptor's iovecs (max_iovecs), rebuilt from its `done` whenever it is prepared (destination). Direct
    // mode: a read lies inside the image and the segments tile it, so it touches at most one run per segment
    // (image_iovecs). The bounce path: one, the read's span of its bounce slot.
    iovecs_.assign(extents * max_iovecs(), iovec{});
    piece_runs_.assign(
        piece_stream_ ? static_cast<size_t>(kBounceSlots * kPieces) * t_.segments.size() : 0, PieceRun{});
    return true;
  }

  // A new read() starts with every descriptor retired and every bank free. This is also what makes a
  // failed call safe to follow: drain() has already retired the kernel's side of everything.
  void reset_pipeline() {
    for (auto& d : descs_)
      d = ExtentDesc{};
    for (auto& r : rows_)
      r = BounceRow{};
    for (int b = 0; b < kBanks; ++b)
      rows_busy_[b] = bank_live_[b] = 0;
  }

  // A row to pack; with piece streaming, a vetted piece not yet handed to a job, whatever its row's state.
  bool has_ready() const {
    for (const auto& r : rows_) {
      if (piece_stream_ ? (r.vetted & static_cast<uint8_t>(~r.dispatched)) != 0 : r.state == RowState::Ready)
        return true;
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
      throw std::runtime_error(error_prefix<Layout>() + "the extent queue overflowed its descriptor count");
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
      if (fault_.poison) poison_slot(slot, kPoisonFill);
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
    if (c.trace)
      c.trace->rows_reading_max = std::max<int64_t>(c.trace->rows_reading_max, static_cast<int64_t>(c.reading_rows));
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
  // kSubReads + k: the stride is the most sub-reads a part can have, and k < sub_reads_per_part <= kSubReads, so the
  // index is unique whatever the row's cut. Credit is untouched: refill() takes it per SQE, so a sub-read costs one
  // like a part did. Pieces with no bytes (past the row's sub-reads) are vetted here, at admission.
  void queue_sub_reads(size_t slot, size_t ordinal, const RowGeometry& g, int64_t admitted) {
    Call& c = c_;
    const size_t parts = static_cast<size_t>(t_.parts);
    const size_t bank = slot / kBounceRows;
    BounceRow& r = rows_[slot];
    r.start = g.start;
    r.subs = static_cast<uint8_t>(g.subs);
    std::copy(g.deps, g.deps + kPieces, r.deps);
    for (int s = 0; s < g.subs; ++s) {
      const uint32_t index =
          static_cast<uint32_t>((slot * parts + static_cast<size_t>(g.part[s])) * kSubReads + g.k[s]);
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
          c.trace->extent_id[trace_slot] = (static_cast<int64_t>(ordinal) << 16) | (static_cast<int64_t>(g.k[s]) << 8) |
                                           static_cast<int64_t>(g.part[s]);
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
  // serving none of a row issues nothing. `prep_read`/`prep_readv` refused (the SQ filled before credit
  // ran out) means the next submit sends what is prepared and refill runs again after the reap. Retries
  // re-enter through the queue, so they take credit like any other read.
  void refill() {
    Call& c = c_;
    int64_t prepared = 0;  // one clock read per refill turn, taken on the first SQE
    while (c.queue_count > 0 && c.pending < c.capacity) {
      const uint32_t index = queue_[c.queue_head];
      ExtentDesc& d = descs_[index];
      const int64_t remaining = d.read->length - d.done;
      const uint64_t tag = (static_cast<uint64_t>(d.generation) << 32) | index;
      const uint64_t offset = static_cast<uint64_t>(d.read->offset + d.done);
      // A resubmission (short read, -EINTR/-EAGAIN) starts past what already landed: the iovecs are rebuilt from
      // `done`, which a mid-file O_DIRECT short read leaves on a block boundary. The kernel is not holding this
      // descriptor's iovecs now: it was either never submitted or its completion was reaped.
      iovec* iov = &iovecs_[static_cast<size_t>(index) * max_iovecs()];
      const unsigned count = destination(d, iov);
      const int fd = fds_[d.read->file];
      // The bounce path keeps IORING_OP_READ (prep_read), the direct path IORING_OP_READV: the default opcodes.
      const bool prepared_one =
          t_.images ? io_.prep_readv(fd, iov, count, offset, tag)
                    : io_.prep_read(fd, iov[0].iov_base, static_cast<unsigned>(iov[0].iov_len), offset, tag);
      if (!prepared_one) break;
      c.queue_head = (c.queue_head + 1) % queue_.size();
      --c.queue_count;
      ++c.pending;
      if (sqe_log_) {
        sqe_log_->push_back(
            SqeRecord{
                d.read->file,
                d.read->offset + d.done,
                remaining,
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
      for (size_t k = 0; k < completions_.size(); ++k)
        process(completions_[k], released);
      if (c.trace) {
        if (c.first_seen == 0) c.first_seen = released;
        c.last_seen = released;
      }
      if (!c.failed) {
        for (uint32_t index : again_)
          queue_push(index);
      }
      return;
    }
    if (c.trace && c.submitted == 0) c.submitted = stamp(c.trace);
    // Fault: reversing only reorders one reaped batch, so wait for every read in flight; otherwise whether
    // anything is reversed depends on how the device happened to batch its completions.
    const unsigned wait_nr = fault_.reverse_cqes && c.pending > 0 ? c.pending : (ready ? 0u : 1u);
    const int rc = submit(wait_nr);
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
    completions_.clear();
    again_.clear();
    const unsigned seen = io_.reap(completions_);
    c.pending -= seen;
    if (fault_.reverse_cqes) std::reverse(completions_.begin(), completions_.end());
    if (fault_.hold_ordinal >= 0) {
      size_t kept = 0;
      for (size_t k = 0; k < completions_.size(); ++k) {
        const uint32_t index = static_cast<uint32_t>(completions_[k].data & 0xFFFFFFFFu);
        const bool live = index < descs_.size() && descs_[index].generation != 0 &&
                          descs_[index].generation == static_cast<uint32_t>(completions_[k].data >> 32);
        if (live &&
            (fault_.hold_rest ? static_cast<int64_t>(rows_[descs_[index].slot].ordinal) >= fault_.hold_ordinal
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
    for (size_t k = 0; k < completions_.size(); ++k)
      process(completions_[k], returned);
    if (c.trace && seen > 0) {
      if (c.first_seen == 0) c.first_seen = returned;
      c.last_seen = returned;
    }
    if (c.failed) return;
    for (uint32_t index : again_)
      queue_push(index);
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
  bool pack_advance() {
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

  // Direct mode's turn of work: publish what has landed (publish_landed).
  bool row_advance() {
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
      if (now < 0) now = stamp(c.trace);
      return now;
    };
    const auto delay = [&] {
      if (fault_.pack_delay_ns <= 0) return;
      std::this_thread::sleep_for(std::chrono::nanoseconds(fault_.pack_delay_ns));
      now = -1;
    };
    if (!piece_stream_) {
      while (true) {
        const size_t best = take_ready_row();
        if (best == static_cast<size_t>(kBounceSlots)) return any;
        delay();
        finish_row(best, clock(), clock());
        any = true;
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
    const int64_t dest = (*c.slots)[rows_[slot].ordinal];
    unsigned count = 0;
    for (const Segment& s : t_.segments) {
      const int64_t lo = std::max(from, s.src), hi = std::min(to, s.src + s.bytes);
      if (lo >= hi) continue;
      out[count++] = iovec{
          t_.slabs[c.layer][s.name] + dest * t_.row_bytes[s.name] + s.dst + (lo - s.src), static_cast<size_t>(hi - lo)};
    }
    return count;
  }

  // The poison fault: fill the memory the row in `slot` is read into, the bounce slot or (direct mode) the
  // destination slab rows, so a byte published without having been read shows.
  void pack_poison_slot(size_t slot, uint8_t fill) {
    std::memset(bounce_slot(slot), fill, static_cast<size_t>(t_.slot_bytes));
  }
  void row_poison_slot(size_t slot, uint8_t fill) {
    const Call& c = c_;
    const int64_t dest = (*c.slots)[rows_[slot].ordinal];
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
  void pack_collect() {
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

  // Direct mode: no piece is ever held by a job (publish_landed publishes on dispatch), so the only collecting left
  // is, with piece streaming, finishing each row whose pieces are all published once its last sub-read retires.
  void row_collect() {
    if (!piece_stream_) return;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      BounceRow& r = rows_[s];
      if (r.state == RowState::Ready && r.published == kAllPieces) {
        finish_row(s, r.pack_first == INT64_MAX ? 0 : r.pack_first, r.pack_last);
      }
    }
  }

  // Wait for every copy the workers still hold, and finish those rows like any other: they were copied
  // whole, and the accounting says so. On a failure the caller still releases every slot; this
  // guarantees nothing writes into them, or reads the bounce, afterwards.
  // With piece streaming every piece job is waited for, then collected and published like any other.
  void pack_quiesce() {
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
  // Direct mode holds no copies: with images c.packing is always 0, so the old body returned at once.
  void row_quiesce() {}

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
      for (const Segment& segment : t_.segments)
        c.trace->useful_bytes += segment.bytes;
      // Rows pack in completion order and may overlap, so the first to finish is not always the first to start.
      if (c.trace->pack_start == 0 || start < c.trace->pack_start) c.trace->pack_start = start;
      c.trace->pack_end = std::max(c.trace->pack_end, end);
      c.trace->pack_ns += end - start;
    }
    if (c.packed) (*c.packed)[ordinal] = 1;
    after_finish(best);
    rows_[best] = BounceRow{};
    --rows_busy_[best / kBounceRows];
  }

  // The poison fault re-fills a finished row's bounce slot, so a later row that reuses it without reading shows.
  void pack_after_finish(size_t slot) {
    if (fault_.poison) poison_slot(slot, kPoisonFill ^ 0xFF);
  }
  // Direct mode: the bytes are the slot's own now, and the caller publishes them.
  void row_after_finish(size_t /*slot*/) {}

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
    return io_.submit(wait_nr);
  }

  // After a failure, empty the ring before the bounce is reused or freed: settle every read prepared or
  // in flight so nothing can still write the bounce once the caller reuses or frees it.
  void drain(unsigned pending) {
    io_.drain(pending);
  }

  Tables t_;
  bool direct_;
  std::vector<int> fds_;
  uint8_t* bounce_ = nullptr;
  std::vector<int64_t> devs_;        // st_dev of each distinct filesystem, in first-opened order
  std::vector<uint8_t> file_drive_;  // per file: its drive slot in a StageRecord
  int64_t drive_dev_[kMaxDrives] = {};
  ReadFault fault_{};
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
  // Declared after pool_ (and so torn down after close(fds_)/std::free(bounce_) in member-destruction
  // order): the workers are joined by the destructor body's own pool_.reset(), before any member
  // destructor runs, so io_'s position here changes nothing about that join. It is safe only because
  // read() always drains the ring (quiesce()) before returning, so nothing is ever in flight when this
  // reader is destroyed.
  Reader io_;
  unsigned configured_queue_depth_ = 0;
  // Indexed by bounce slot with the flag off, by (slot, piece) with piece streaming (size_jobs).
  PackJob jobs_[kBounceSlots * kPieces];
  std::vector<CopyRun> runs_;
  // Piece streaming (set_piece_stream; off by default). subs_ is the most sub-reads per part: 1 with the flag off,
  // which makes descriptor (slot, part, sub) the old (slot, part), and kSubReads with it on whatever a row's cut
  // (row_geometry cuts each reading part into sub_reads_per_part <= kSubReads). sub_reads_ holds each live sub-read's
  // Read (a descriptor points into it), piece_runs_ each slot's piece runs, geometry_ a batch's rows between
  // validation and admission. All sized at open() or set_piece_stream(), and empty with the flag off.
  bool piece_stream_ = false;
  size_t subs_ = 1;
  std::vector<Read> sub_reads_;
  std::vector<PieceRun> piece_runs_;
  std::vector<iovec> iovecs_;  // direct mode: segments.size() per descriptor (size_extents)
  RowGeometry geometry_[kBounceRows];
  std::vector<SqeRecord>* sqe_log_ = nullptr;  // test only (set_sqe_log)
  int64_t publishes_ = 0;          // pieces published over the reader's life (the publish_twice fault counts them)
  int64_t publish_refused_ = 0;    // publish attempts a readiness word refused
  size_t rows_busy_[kBanks] = {};  // rows not yet packed, per bank: the packing references
  size_t bank_live_[kBanks] = {};  // extents not yet retired, per bank: the I/O references
  uint32_t generation_ = 0;
  int64_t generation_wraps_ = 0;
  int64_t stale_cqes_ = 0;
  int64_t retired_ = 0;
  Completion stale_{0, 0};
  uint32_t stale_index_ = 0;
  bool stale_waiting_ = false;  // a retired completion is held until its descriptor is recycled
  bool stale_armed_ = false;    // ... and has been: deliver it with the next reap
};

}  // namespace expert_stream
}  // namespace sglang
