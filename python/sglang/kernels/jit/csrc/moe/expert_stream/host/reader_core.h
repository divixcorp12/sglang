// ReaderCore: the io_uring read pipeline of the expert-stream host reader.
//
// ReaderCore reads whole expert rows (aligned supersets of the bytes a row needs) from the mirror files into the slab
// rows, through per-extent descriptors, ring credit and banks. RowReader is the one derived reader: it reads row
// images with O_DIRECT straight into the slab rows. The pipeline is written once here; the derived reader supplies
// the destination and publishing hooks.
//
//   NoProgress    read()'s default progress callback (compiled out)
//   SqeRecord     test-only log entry of one prepared SQE
//   ReaderCore    the pipeline: admit batches, prepare SQEs under credit, reap, retire, vet and publish
//
// Build policy (build_policy.h): ProdBuild has no metrics, no fault state and no trace record, so the request path
// tests none of them; InstrBuild adds them for tests (read_fault.h, StageRecord in reader_base.h).
#pragma once

#include "../row_layout.h"
#include "build_policy.h"
#include "drive_load.h"
#include "file_reader.h"
#include "piece_geometry.h"
#include "read_cuts.h"
#include "uring_options.h"
#include <cstdio>
#include <memory>
#include <span>
#include <type_traits>

namespace sglang {
namespace expert_stream {

// read()'s default progress callback: does nothing. A distinct type, so read() compiles the call out rather than
// testing it.
struct NoProgress {
  void operator()() const {}
};

// read()'s default root policy: every sub-read reads the root its table (or the dynamic choice, set_mirror_caps) gives
// it, and as many go in flight as credit allows. A distinct type, so read() compiles the gated paths out.
struct TableRoots {
  static constexpr bool kGated = false;
};

// A speculative read's root policy (SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE; design 2026-10-09-dsv41-drive-aware-reads,
// change (2)), passed to read() by the speculative reader. The row reads one piece at a time, all from one root
// with no demand sub-read in flight on any reader of the drive load: the first piece picks the demand-free root with
// the fewest bytes in flight, and each later piece rechecks it and moves to another demand-free root when demand
// arrived. With none demand-free the piece waits ("defers") without a ring wait; waiting past `deadline_ns`, or
// `give_up` saying so, abandons the read (read() returns 0, `abandoned` set). `boost` set (a forced miss waits on
// this row) drops all of that for the rest of the row: its pieces read their table roots, as many as credit allows.
// Needs the mirror file map (set_mirror_map).
struct IdleRoots {
  static constexpr bool kGated = true;
  const std::atomic<uint32_t>* boost = nullptr;  // nonzero: boosted (null: never)
  int64_t deadline_ns = 0;                       // the longest wait for an idle root
  bool (*give_up)(void*) = nullptr;              // true abandons the read before its next piece (null: never)
  void (*on_piece)(void*, int root) = nullptr;   // after each piece is prepared (null: nothing)
  void* context = nullptr;                       // give_up's and on_piece's argument
  // What the read did, for the caller's counters.
  int64_t deferrals = 0;  // waits for an idle root (one per wait, however long)
  int64_t moves = 0;      // pieces sent to another root than the row's earlier pieces
  bool boosted = false;
  bool abandoned = false;
};

// Test only: one prepared SQE, as ReaderCore::set_sqe_log records it.
struct SqeRecord {
  int64_t file, offset, length, bounce;  // bounce: byte offset of the destination from the bounce's start
};

// The read pipeline shared by the derived readers.
//
// Terms. "Bank", "bounce slot" and "packing" name pipeline state (a row's descriptors, its bank's references, its
// finish), not memory: RowReader allocates no bounce buffer and copies nothing, because O_DIRECT reads land in the slab
// rows themselves. The base/derived split separates this pipeline from the destination and publishing hooks.
//
// Pipeline. A read() call is split into batches of `step` rows; batch b fills bank b % kBanks. Every bounce slot is
// one row's aligned superset, and every extent has its own preallocated descriptor. Descriptor (slot, part) is the
// SQE's user_data together with a generation, so a completion can only be attributed to the extent that is live in
// that descriptor NOW. Three resources are independent of each other:
//   * ring credit    `pending <= capacity` (queue_depth()): how many SQEs may be prepared and not yet reaped. It knows
//                    nothing about banks; a bank can hold more extents than the ring.
//   * banks          kBanks * kBounceRows slots. A bank is handed to a new batch only once every row of its previous
//                    batch has PACKED (and so every extent has completed and retired). I/O and packing are the two
//                    references a bank holds, and both must be gone before the kernel may write into it again.
//   * reading rows   at most `max_reading_rows` rows with I/O outstanding.
//
// Packing and publishing. A row is packed as soon as ITS extents have completed, while other rows are still in
// flight, and every completed row is packed before read() returns. read() itself publishes no row: the caller keeps
// the slots unmapped until read() returns 1, so no row is visible before the whole request is. With piece streaming
// each vetted piece is packed by its own job, and the owner publishes it into the caller's readiness words
// (PiecePublish) once that job is done; the slot map is still the caller's.
//
// Failure. Packing writes only into the caller's not-yet-published slots and only from a slot whose extents have all
// completed, so a failure leaves at most fully packed rows in unpublished slots, never a half-packed one, and the
// caller releases them. With piece streaming the unit is the piece: a piece is packed only once the sub-reads it
// depends on have landed and is published only once its job is done, so a failure leaves whole published pieces, never
// a torn one, in slots whose map the caller has not published. The RAM tier fail-stops on a failed demand read
// (RamTier::fail_record), and a failed prefill fill releases the slots it did not finish packing. Every return of
// read() leaves the ring empty: nothing in flight, nothing prepared.
//
// Threading. All members are used by one owner thread (the RAM-miss service thread, or the test's caller).
//
// Derived hooks. Where the readers differ, the pipeline calls a hook on `Derived` (derived().X()):
// check_piece_stream_support, on_piece_stream_set, open_memory, open_workers, registered_regions, max_iovecs,
// destination, advance, collect, quiesce, poison_slot and after_finish, each private to the derived reader, which
// befriends this class. `Derived::kScatter` picks the read opcode. Derived readers are never deleted through this
// class, so its destructor is protected and not virtual.
//
// `Build` (ProdBuild or InstrBuild, build_policy.h) is explicit, never read from `Derived`: `Derived` is incomplete
// while this base is instantiated, and the derived reader forwards the same `Build` it was given.
template <class Derived, ExpertRowLayout Layout, AsyncFileReader Reader, class Build>
class ReaderCore {
  static_assert(BuildPolicy<Build>);
  // Faults lean on metrics (the cqe_call fault counts through metric(&ReaderMetrics::cqes)): a build with faults and no
  // metrics would inject at the wrong completion, silently.
  static_assert(!Build::kFaults || Build::kMetrics, "a build with faults needs metrics");

 public:
  using LayoutType = Layout;  // named so it cannot shadow the template parameter
  using BuildType = Build;
  using SqeRecord = expert_stream::SqeRecord;

  ReaderCore(const ReaderCore&) = delete;  // owns fds and the ring
  ReaderCore& operator=(const ReaderCore&) = delete;

  const Tables& tables() const {
    return t_;
  }

  // Turns piece streaming on or off (sub_reads_per_part(reading) sub-reads per reading part, per-piece vetting,
  // packing and publishing); off by default. Call it before open(), or on an idle reader after it (the tier sets it
  // before its service thread starts), since it resizes the descriptor arrays. Throws with more mirror parts than
  // pieces (a reading part needs a piece of its own), or when a slab row base is not kPieceAlign-aligned (a piece's
  // cuts are aligned in the row).
  void set_piece_stream(bool on) {
    if (on) {
      derived().check_piece_stream_support();
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
    derived().on_piece_stream_set();
  }

  bool piece_stream() const {
    return piece_stream_;
  }

  // Where this reader counts its reads in flight per root (drive_load.h): the tier points every reader at its one
  // DriveLoad; a reader on its own (tests, tools) counts into its own. Call it on an idle reader.
  void set_drive_load(DriveLoad* load) {
    load_ = load != nullptr ? load : &own_load_;
  }
  const DriveLoad& drive_load() const {
    return *load_;
  }

  // The kind the next read()s count as (kDemandRead, the default, or kSpecRead). The tier's speculative read sets it
  // for its one read and clears it, on the thread that holds the reader. Call it on an idle reader.
  void set_read_kind(int kind) {
    kind_ = kind;
  }

  // SGLANG_MOE_EXPERT_MIRROR_DYNAMIC (design 2026-10-09-dsv41-drive-aware-reads, change (3)): `caps`, one positive
  // in-flight cap per mirror root, turns the dynamic root choice on (choose_root); empty turns it off, and the reader
  // then reads every sub-read from its own root, as before. Piece streaming only (set it first). Builds the mirror file
  // map (set_mirror_map). Call it on an idle reader.
  void set_mirror_caps(std::span<const int64_t> caps) {
    const std::string prefix = error_prefix<Layout>() + "SGLANG_MOE_EXPERT_MIRROR_DYNAMIC: ";
    if (caps.empty()) {
      dynamic_ = false;
      alt_file_.clear();
      return;
    }
    const int64_t parts = t_.parts;
    if (static_cast<int64_t>(caps.size()) != parts) {
      throw std::runtime_error(
          prefix + "needs one in-flight cap per mirror root: " + std::to_string(caps.size()) + " caps for " +
          std::to_string(parts) + " mirror roots");
    }
    for (int64_t cap : caps)
      if (cap < 1) throw std::runtime_error(prefix + "every in-flight cap must be positive, got " + std::to_string(cap));
    build_mirror_map(prefix);
    for (int64_t q = 0; q < parts; ++q)
      caps_[q] = caps[static_cast<size_t>(q)];
    dynamic_ = true;
  }

  // Builds the mirror file map a sub-read needs to read another root's copy: the dynamic choice (set_mirror_caps) and
  // the speculative reader's IdleRoots. Every root holds every row's image at the same offsets, so a sub-read can read
  // any root's file of its row; this refuses the tables when any row's files are not a full mirror set (the same source
  // and size on every root, the root being file % parts). Piece streaming only (set it first). Call it on an idle
  // reader.
  void set_mirror_map() {
    build_mirror_map(error_prefix<Layout>() + "the mirror file map: ");
  }

  // Test only: the sub-reads this reader counts in flight now, its share of the DriveLoad (0 between reads).
  int64_t drive_share_reads() const {
    int64_t reads = 0;
    for (const auto& root : mine_)
      for (const auto& kind : root)
        reads += kind.reads;
    return reads;
  }

  // Pieces a readiness word refused to publish, over the reader's life (each also failed its read). A metric: 0 in
  // ProdBuild.
  int64_t publish_refused() const {
    return metric(&ReaderMetrics::publish_refused);
  }

  // Test only: the descriptor count and the ring credit this reader runs with.
  size_t descriptors() const {
    return descs_.size();
  }

  unsigned credit() const {
    return queue_depth();
  }

  // Test only, InstrBuild only: appends every SQE prepared to `log` while set (null: nothing recorded, one branch per
  // SQE).
  void set_sqe_log(std::vector<SqeRecord>* log)
    requires(Build::kFaults)
  {
    faults_.sqe_log = log;
  }

  // Test-only scaffold: pins the owner thread to `core` at open() (-1, the default, leaves it unpinned). Kept from the
  // owner-pinning measurement in analysis/dsv41-drive/PACK_WORKERS.md, where it kept the owner off a packing pool.
  void set_owner_core(int64_t core) {
    owner_core_ = core;
  }

  // Test only, InstrBuild only: arms `fault` (read_fault.h) and resets its per-fault state. ProdBuild has no fault
  // state at all.
  void set_fault(const ReadFault& fault)
    requires(Build::kFaults)
  {
    faults_.fault = fault;
    faults_.part_fired = false;
    faults_.retired = 0;
    faults_.stale_armed = false;
    faults_.stale_waiting = false;
    if (fault.generation_start != 0) generation_ = static_cast<uint32_t>(fault.generation_start);
    if constexpr (requires { io_.set_submit_fault(SubmitFault{}); }) {
      io_.set_submit_fault(
          SubmitFault{
              fault.submit_error,
              fault.submit_call,
              fault.submit_first,
              fault.submit_short_call,
              fault.ring_reset_fail,
              fault.nop_flush_refused});
    } else if (
        fault.submit_error != 0 || fault.submit_call != 0 || fault.submit_first || fault.submit_short_call != 0 ||
        fault.ring_reset_fail || fault.nop_flush_refused) {
      // Fallback for a Reader that cannot inject submit faults. The instrumented instantiation
      // (exl3_ram_miss_host_instr.cpp) pairs RowReader with FaultyReader<InstrUringReader>, which can, so no test
      // reaches this.
      throw std::runtime_error(error_prefix<Layout>() + "this reader cannot inject submit faults");
    }
  }

  // Completions reaped over the reader's life (tests: a zero-length extent must add none). The reader's diagnostic
  // counters are metrics: every getter below returns 0 in ProdBuild.
  int64_t cqes() const {
    return metric(&ReaderMetrics::cqes);
  }

  // Completions that named no live descriptor, and generation counter wraps (tests).
  int64_t stale_cqes() const {
    return metric(&ReaderMetrics::stale_cqes);
  }

  int64_t generation_wraps() const {
    return metric(&ReaderMetrics::generation_wraps);
  }

  // Most legs a read may have: the width of the completion tag's leg field (make_tag). A fixed read (READ_MODE
  // fixed/readv_fixed) whose iovecs lie in k registered buffers is prepared as k legs, one SQE each, submitted
  // together; with read cuts a read is also cut into device-sized legs (read_cuts.h). Legs per read are sized at open()
  // (leg_stride_).
  static constexpr unsigned kMaxLegs = 255;

  // Test only (fault word fixed_chunk_cap): sets the registration chunk cap before open() (0: the 1 GiB default).
  void set_fixed_chunk_cap(int64_t cap) {
    if constexpr (requires(Reader& reader) { reader.set_fixed_chunk_cap(size_t{0}); }) {
      io_.set_fixed_chunk_cap(static_cast<size_t>(std::max<int64_t>(0, cap)));
    } else if (cap != 0) {
      throw std::runtime_error(error_prefix<Layout>() + "this reader registers no buffers");
    }
  }

  // Logical reads prepared as more than one leg, and the SQEs those reads issued (first attempts), over the life.
  int64_t fixed_cuts() const {
    return metric(&ReaderMetrics::fixed_cuts);
  }

  int64_t fanout_sqes() const {
    return metric(&ReaderMetrics::fanout_sqes);
  }

  // Test only (fault word leg_cut_cap): cuts every read at `cap` bytes (whole pages) on a 4 KiB boundary, whatever
  // READ_CUTS says. Call it before open(). 0 restores READ_CUTS and the device limits.
  void set_leg_cut_cap(int64_t cap) {
    leg_cut_cap_ = std::max<int64_t>(0, cap);
  }
  // Reads planned as more than one cut run, and the runs a boundary gap opened (first plans).
  int64_t cut_reads() const {
    return metric(&ReaderMetrics::cut_reads);
  }
  int64_t gap_cuts() const {
    return metric(&ReaderMetrics::gap_cuts);
  }

  // Legs a read may have (the storage stride) and the smallest cut in force (0: cuts off).
  int64_t leg_stride() const {
    return leg_stride_;
  }
  int64_t min_cut_bytes() const {
    if (!cuts_) return 0;
    int64_t least = INT64_MAX;
    for (const auto& l : limits_)
      least = std::min(least, l.cut_bytes);
    return least;
  }

  // Before open(): the ring's SQPOLL core (BasicUringReader::set_sq_thread_cpu).
  void set_sq_thread_cpu(int cpu) {
    io_.set_sq_thread_cpu(cpu);
  }

  // Opens the files, sizes every buffer and initializes the ring. Returns false (after logging) on an open or init
  // failure; throws on a file whose size differs from its source. Allocates; call once, on the owner thread.
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
      // The table clamps every read at end of file against the SOURCE size (file_sizes), so a copy of another size
      // would otherwise be clamped, or over-read, into a short or stale row that looks complete. Name both files:
      // with dozens of shards a bare "size mismatch" does not say which copy is bad.
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
    if (!derived().open_memory()) return false;
    size_legs();
    if (cuts_) {  // one line per drive: the cut in force and where it came from (a fallback names its reason)
      std::vector<bool> said(devs_.size(), false);
      for (size_t f = 0; f < fds_.size(); ++f) {
        const size_t drive = file_drive_[f];
        if (drive >= said.size() || said[drive]) continue;
        said[drive] = true;
        const auto dev = static_cast<dev_t>(devs_[drive]);
        std::fprintf(
            stderr,
            "expert stream io_uring: read cuts: drive=%zu dev=%u:%u cut_bytes=%lld virt_mask=%llu source=%s\n",
            drive,
            major(dev),
            minor(dev),
            static_cast<long long>(limits_[f].cut_bytes),
            static_cast<unsigned long long>(limits_[f].virt_mask),
            limits_[f].source.c_str());
      }
    }
    if (!size_extents()) return false;
    if (!io_.init(queue_depth())) return false;
    // One reap returns at most a queue depth of CQEs (plus a withheld stale one): reserve that here, so the widest reap
    // never grows the list on the service thread. This supersedes size_extents' smaller `extents + 1`.
    completions_.reserve(queue_depth() + 1);
    // A read's legs are reserved all at once, so the ring must hold the widest read's legs (leg_stride_ bounds both a
    // fixed read's leg per registered buffer and a cut read's device-sized legs).
    if (fixed_reads() && leg_stride_ < std::max<size_t>(1, derived().max_iovecs()))
      throw std::logic_error(error_prefix<Layout>() + "a fixed read mode opened with one-leg storage");
    if (cuts_ && configured_queue_depth_ != 0) {
      const unsigned scaled = static_cast<unsigned>(
          std::min<size_t>(32768, static_cast<size_t>(kQueueDepth) * static_cast<size_t>(t_.parts) * leg_stride_));
      if (configured_queue_depth_ < scaled)
        std::fprintf(
            stderr,
            "expert stream io_uring: read cuts: SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=%u is below the cut default %u "
            "(16 x %zu parts x %u legs): fewer reads fit in flight than uncut\n",
            configured_queue_depth_,
            scaled,
            static_cast<size_t>(t_.parts),
            leg_stride_);
    }
    if ((fixed_reads() || cuts_) && queue_depth() < leg_stride_) {
      throw std::runtime_error(
          error_prefix<Layout>() + "reads fanned out or cut into legs need a queue depth of at least " +
          std::to_string(leg_stride_) +
          " (the widest read's legs); SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=" + std::to_string(queue_depth()));
    }
    if constexpr (requires(
                      Reader& reader, const std::vector<int>& files, const std::vector<RegisteredRegion>& buffers) {
                    reader.configure_resources(files, buffers, true);
                  }) {
      io_.configure_resources(fds_, derived().registered_regions(), direct_);
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
    derived().open_workers(inherited);
    return true;
  }

  // Reads `experts` of streamed row `layer` into `slots`, `step` rows per io_uring batch (at most kBounceRows, one
  // bank), on the owner thread. Allocates nothing.
  //
  // Returns 1 when every row landed, 0 on an I/O error or short file, -1 when abandoned before every batch was
  // admitted. Every return leaves the ring empty: nothing in flight, nothing prepared.
  //
  //   abandon           `abandon(batches admitted so far)` runs before each batch is admitted and whenever the loop
  //                     comes back to it; true stops admitting new batches. Rows already admitted are reaped and
  //                     packed.
  //   trace             when not null, receives this read's stage stamps and per-drive bytes (StageRecord). Null costs
  //                     a branch per event and no clock read; stamps only read the clock and add to `trace`, so they
  //                     cannot change what is submitted, reaped, drained or copied.
  //   packed            when not null, set to 1 for every row that was packed. With -1 those rows are complete and the
  //                     caller may keep them; with 0 the caller releases everything.
  //   max_reading_rows  caps the rows with I/O outstanding.
  //   progress          unless it is NoProgress, invoked once per turn of the drain loop and again as each row finishes
  //                     (finish_row), so a caller that publishes the landed prefix (run_fill's advance) sees every row
  //                     as it lands, not once per turn: one reap can return every row of a read at once. The reader has
  //                     no lease vocabulary; this callback is how the caller (serve()) runs its own periodic work
  //                     (retire_leases()) while a read is in flight, as `abandon` is how it decides when to stop
  //                     admitting. NoProgress compiles to nothing.
  //   publish           piece streaming only: where the owner publishes each piece (PiecePublish).
  //   roots             the root policy: TableRoots (the default, every demand read), or a speculative read's IdleRoots
  //                     (piece streaming and set_mirror_map only), which reads one piece at a time from a root no
  //                     demand reads, waits for one, and returns 0 with `roots.abandoned` set when it gives up.
  //
  // `abandon` and `progress` are template parameters, not std::function: a closure past std::function's local storage
  // would be heap-allocated per read. `experts` and `slots` are spans, so the service passes its fixed-size lists
  // without copying them into vectors.
  //
  // This is the hot path and checks nothing: `layer`, `experts` and `slots` must be in range and
  // `experts.size() == slots.size()`. The service and read_rows_once (Python) validate at their boundaries.
  template <class Abandon, class Progress = NoProgress, class RootPolicy = TableRoots>
  int read(
      int64_t layer,
      std::span<const int32_t> experts,
      std::span<const int64_t> slots,
      size_t step,
      Abandon&& abandon,
      StageRecord* trace = nullptr,
      std::vector<uint8_t>* packed = nullptr,
      size_t max_reading_rows = SIZE_MAX,
      Progress&& progress = Progress{},
      const PiecePublish* publish = nullptr,
      RootPolicy&& roots = RootPolicy{}) {
    if (!io_.ready()) return 0;
    if constexpr (std::decay_t<RootPolicy>::kGated) {
      if (!piece_stream_ || alt_file_.empty())
        throw std::logic_error(error_prefix<Layout>() + "a gated read needs piece streaming and the mirror file map");
    }
    step = std::max<size_t>(1, std::min<size_t>(step, kBounceRows));
    Call& c = c_;
    c = Call{};
    c.layer = layer;
    c.experts = experts;
    c.slots = slots;
    c.step = step;
    c.total = experts.size();
    c.batches = (c.total + step - 1) / step;
    c.max_reading = std::max<size_t>(1, max_reading_rows);
    // Credit is the ring's alone: banks and rows in flight do not enter it.
    c.capacity = queue_depth();
    if constexpr (Build::kFaults) {
      if (faults_.fault.max_outstanding > 0)
        c.capacity = std::min<unsigned>(c.capacity, static_cast<unsigned>(faults_.fault.max_outstanding));
    }
    if constexpr (Build::kMetrics) c.trace = trace;  // ProdBuild: no record, whatever the caller passed
    c.packed = packed;
    if constexpr (!std::is_same_v<std::decay_t<Progress>, NoProgress>) {
      // Type-erased as a function pointer and the closure's address: nothing allocated. The closure outlives the call
      // (it is read()'s own parameter), and finish_row runs only inside the loop below.
      c.progress = [](void* closure) { (*static_cast<std::remove_reference_t<Progress>*>(closure))(); };
      c.progress_closure = const_cast<void*>(static_cast<const void*>(std::addressof(progress)));
    }
    c.publish = piece_stream_ ? publish : nullptr;
    if (packed) packed->assign(c.total, 0);
    on_trace([&](StageRecord& t) {
      t.rows_asked = static_cast<int64_t>(c.total);
      t.pack_workers = derived().pack_workers();
      t.pack_split = derived().pack_split();
      t.piece_stream = piece_stream_ ? 1 : 0;
    });
    reset_pipeline();
    if constexpr (Build::kFaults) faults_.held.clear();
    // However this call ends, nothing may still be reading or copying: the caller releases the slots on return and
    // the next read reuses them. An exception (one of the accounting guards) leaves reads in flight, and in direct mode
    // those reads write the slab rows themselves, so they are drained before unwinding past the caller.
    // "Unwinding" is every exit but the two returns below, which set `returned` just before returning. This is not
    // std::uncaught_exceptions(): its first call on a thread makes __tls_get_addr allocate libstdc++'s exception
    // globals (dynamic TLS), which was the production service thread's one malloc.
    struct Quiesce {
      ReaderCore* reader;
      bool returned = false;
      ~Quiesce() {
        reader->derived().quiesce();
        if (!returned) {
          // Already unwinding: a failed ring reset here leaves nothing in flight (it waits for every consumed SQE
          // first, and the reader is then closed), so the exception in flight is the one the caller should see.
          try {
            reader->drain(reader->c_.pending);
          } catch (const std::exception&) {
          }
        }
      }
    } quiesce_on_exit{this};
    while (true) {
      // Progress runs every turn: the caller's hook (retire_leases) returns at once when no lane is outstanding, and
      // gating it on a clock would put a clock read on every turn of the hot path. It is not gated on a completion
      // being reaped either: those are this reader's own I/O, uncorrelated with the device acknowledging a lease.
      if constexpr (!std::is_same_v<std::decay_t<Progress>, NoProgress>) progress();
      derived().collect();  // before admit: a bank whose last copy just finished is free for the next batch
      if (!c.failed) admit(abandon);
      if (!c.failed) refill(roots);
      if (c.failed) break;
      const bool ready = has_ready();
      if (c.pending == 0 && !ready && held_empty() && c.packing == 0) {
        if constexpr (std::decay_t<RootPolicy>::kGated) {
          // A piece waits for an idle root (refill deferred it) with nothing in flight: no ring wait, poll again.
          if (c.queue_count > 0) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(kDeferSleepNs));
            continue;
          }
        }
        break;
      }
      // Submit what was prepared before packing, so storage stays busy while the CPU copies; block for a completion
      // only when there is no complete row to pack. With jobs packing on workers the owner cannot be woken from a
      // blocking wait when one finishes, so it polls instead. The slow-drive fault's withheld completions arrive only
      // once every other row has packed.
      if (c.pending > 0 || (!ready && c.packing == 0 && !held_empty())) reap(ready || c.packing > 0);
      if (c.failed) break;
      if (!derived().advance() && c.packing > 0) _mm_pause();
    }
    // Nothing in flight and nothing to pack, yet a row was left unread or unpacked, or a batch was neither admitted nor
    // abandoned: the loop's own bookkeeping is wrong. Fail rather than return a row that was never read.
    if (!c.failed) {
      bool clean = c.reading_rows == 0 && c.queue_count == 0;
      for (int b = 0; b < kBanks; ++b)
        clean = clean && rows_busy_[b] == 0 && bank_live_[b] == 0;
      if (!clean || (!c.abandoned && c.next_batch < c.batches)) c.failed = true;
    }
    on_trace([&](StageRecord& t) {
      t.submit = c.submitted;
      if (c.first_seen != 0) {
        t.first_cqe = c.first_seen;
        t.last_cqe = c.last_seen;
        t.submit_to_first_cqe_ns = c.first_seen - c.submitted;
        t.first_to_last_cqe_ns = c.last_seen - c.first_seen;
      }
    });
    if (c.failed) {
      account_unfinished();
      drain(c.pending);
      quiesce_on_exit.returned = true;
      return 0;
    }
    quiesce_on_exit.returned = true;
    return c.abandoned && c.next_batch < c.batches ? -1 : 1;
  }

 protected:
  // `direct` opens the files with O_DIRECT. Queue depth, read cuts and the read mode come from
  // UringOptions::from_env().
  ReaderCore(Tables tables, bool direct) : t_(std::move(tables)), direct_(direct) {
    configured_queue_depth_ = UringOptions::from_env().queue_depth;
    cuts_requested_ = UringOptions::from_env().read_cuts_on();
    fixed_requested_ = UringOptions::from_env().read_mode != UringReadMode::Normal;
  }

  ~ReaderCore() {
    // Registered regions must be released while their allocations and files still exist.
    close_io();
    for (int fd : fds_)
      ::close(fd);
    // Undo the owner-pin scaffold's affinity change: the pin targets the calling thread, which a caller (e.g. the
    // benchmark) may reuse across many readers, so a later open() must see the original mask, not the single core this
    // reader pinned itself to.
    if (owner_pinned_) pthread_setaffinity_np(pthread_self(), sizeof(unpinned_affinity_), &unpinned_affinity_);
  }

  // Release what the ring registered (Reader::close, when the reader has one). Idempotent: a derived destructor
  // calls it before freeing registered memory, and ~ReaderCore calls it again.
  void close_io() {
    if constexpr (requires(Reader& reader) { reader.close(); }) io_.close();
  }

  Derived& derived() {
    return static_cast<Derived&>(*this);
  }
  const Derived& derived() const {
    return static_cast<const Derived&>(*this);
  }

  static constexpr int kMaxSoftErrors = 1000;         // consecutive -EINTR/-EAGAIN/-EBUSY submits before the read fails
  static constexpr int kMaxRetries = 8;               // resubmissions of one descriptor before the read fails
  static constexpr uint8_t kPoisonFill = 0xA5;        // the `poison` fault's bounce-slot fill
  static constexpr int32_t kPoisonSlot = 0x7EADBEEF;  // the `poison` fault's scribble over a retired descriptor's slot
  static constexpr int64_t kDeferSleepNs = 10'000;    // IdleRoots: a deferred read's poll interval

  // A bounce row's life: Free, then Reading until its last extent retires, then Ready to pack. Packing means handed to
  // a packing worker, which owns the copy until the owner sees its job done.
  enum class RowState : uint8_t { Free, Reading, Ready, Packing };

  // One extent's read, live from admission until its last completion retires it (generation != 0). The descriptor's
  // index is (slot * parts + part) * subs_ + sub, and rides in the SQE's user_data (make_tag).
  struct ExtentDesc {
    const Read* read = nullptr;
    int64_t done = 0;
    int64_t expected = 0;
    uint32_t generation = 0;  // 0: retired, no completion may name it
    int32_t retries = 0;
    int32_t slot = -1;          // bounce slot: bank * kBounceRows + row within the bank
    int32_t trace_slot = -1;    // index into the record's extent arrays, -1 when not stamped
    int32_t sub = 0;            // piece streaming: the sub-read's ordinal in its row's file order
    uint8_t legs = 0;           // planned legs (legs_[index * leg_stride_ ...]); 0: not yet prepared
    uint8_t legs_inflight = 0;  // legs prepared and not yet reaped
    bool queued = false;        // in queue_ (or about to be, through again_): a descriptor is queued at most once
  };

  // The state of one leg: Idle (to be prepared, or to be resubmitted after a short or retried read), Inflight (an SQE
  // is prepared and not reaped) or Done.
  enum class LegState : uint8_t { Idle, Inflight, Done };

  // One leg of a descriptor's read: the SQE unit. Default reads are one leg covering the whole read; a fixed read has
  // a leg per registered buffer its iovecs meet (UringReader::fixed_legs), and a cut read a leg per device-sized run.
  struct Leg {
    int64_t start = 0;     // the leg's first byte, from the read's start (d.read->offset / dest)
    int64_t bytes = 0;     // what the leg covers
    int64_t expected = 0;  // bytes, clamped at the read's end-of-file expectation (d.expected)
    int64_t done = 0;
    unsigned first_iov = 0, iov_count = 0;
    int buffer = -1;  // registered buffer; -1 outside fixed modes
    LegState state = LegState::Idle;
  };

  // One row's slot in a bank: its state, its coverage accounting and, with piece streaming, its sub-read and piece
  // bookkeeping.
  struct BounceRow {
    RowState state = RowState::Free;
    size_t ordinal = 0;  // the row's index in the request
    unsigned extents_left = 0;
    // Coverage, checked before the row is packed. `needed` is the last byte of the slot the segments can read; `filled`
    // is what the drives actually delivered into it. See take_ready_row().
    int64_t needed = 0;
    int64_t filled = 0;
    // Piece streaming only (zero otherwise): the row's sub-reads by ordinal and its pieces. A piece is vetted once
    // every sub-read in its dependency mask has landed and its bytes lie inside what they delivered; with the flag on
    // this, not `filled`, is what coverage rests on.
    int64_t start = 0;  // where the needed bytes begin in the slot
    uint8_t subs = 0;
    uint8_t landed = 0;  // bit s: sub-read s retired
    uint8_t vetted = 0;  // bit j: piece j vetted
    uint8_t deps[kPieces] = {};
    int64_t sub_dest[kPieces] = {};
    int64_t sub_done[kPieces] = {};
    // Bit j: piece j handed to a packing job (or, with no bytes, published at once) / collected and published by the
    // owner. The row is finished once every piece is published and every sub-read retired.
    uint8_t dispatched = 0;
    uint8_t published = 0;
    int64_t pack_first = INT64_MAX;  // the earliest start and latest end of its pieces' jobs (traced reads only)
    int64_t pack_last = 0;
  };

  using Completion = ReadCompletion;

  // The reader's diagnostic counters over its life (InstrBuild only). The Call fields `first_seen`, `last_seen` and
  // `submitted` stay plain in both builds: only the trace sets them, and they cost a store, not a branch.
  struct ReaderMetrics {
    int64_t cqes = 0;              // completions reaped
    int64_t stale_cqes = 0;        // completions that named no live descriptor
    int64_t generation_wraps = 0;  // generation counter wraps
    int64_t cut_reads = 0;         // reads planned as more than one cut run (first plans)
    int64_t gap_cuts = 0;          // runs a boundary gap opened (first plans)
    int64_t fixed_cuts = 0;        // reads prepared as more than one leg (first attempts)
    int64_t fanout_sqes = 0;       // the SQEs those reads issued (first attempts)
    int64_t publish_refused = 0;   // publish attempts a readiness word refused
  };
  struct NoReaderMetrics {};
  // Test-only fault state (set_fault, set_sqe_log), InstrBuild only, so ProdBuild's request path tests none of it.
  struct FaultState {
    ReadFault fault{};
    bool part_fired = false;
    int64_t retired = 0;  // extents retired (the stale_cqe_call fault counts them)
    Completion stale{0, 0};
    uint32_t stale_index = 0;
    bool stale_waiting = false;  // a retired completion is held until its descriptor is recycled
    bool stale_armed = false;    // ... and has been: deliver it with the next reap
    int64_t publishes = 0;       // pieces published over the reader's life (the publish_twice fault counts them)
    std::vector<SqeRecord>* sqe_log = nullptr;  // set_sqe_log
    std::vector<Completion> held;               // completions withheld from the reader (hold_ordinal)
  };
  struct NoFaultState {};

  // The stage trace: ProdBuild's Call carries no record pointer at all, so no site can test or stamp one.
  struct NoTrace {};
  using TracePtr = std::conditional_t<Build::kMetrics, StageRecord*, NoTrace>;

  // One read() call's state. Everything the pipeline mutates lives here or in the members below, all sized at open();
  // read() allocates nothing.
  struct Call {
    int64_t layer = 0;
    std::span<const int32_t> experts;
    std::span<const int64_t> slots;
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
    [[no_unique_address]] TracePtr trace{};  // ProdBuild: an empty member, so no site can test or stamp it
    std::vector<uint8_t>* packed = nullptr;
    void (*progress)(void*) = nullptr;  // read()'s progress hook, also run as each row finishes (null: NoProgress)
    void* progress_closure = nullptr;
    bool failed = false;
    bool abandoned = false;
    bool stalled = false;  // a batch is waiting for its bank to retire
    size_t packing = 0;    // jobs handed to the packing workers and not yet collected by the owner (rows, or pieces)
    const PiecePublish* publish = nullptr;  // piece streaming: where the owner publishes (null: nowhere)
    int soft_errors = 0;
    int64_t submitted = 0, first_seen = 0, last_seen = 0;
    int64_t events = 0;    // piece streaming: sub-read landings and piece vettings so far (the trace's sequence)
    size_t published = 0;  // piece streaming: pieces published so far (the last_publish_delay_ns fault)
    int reads_inflight = 0;      // sub-reads with a leg in flight (count_drive)
    int gate_root = -1;          // IdleRoots: the root the row's last piece read (-1: none yet)
    int64_t deferred_since = 0;  // IdleRoots: when the piece at the queue head began waiting for an idle root (0: not)
  };

  // Runs `f` on this read's stage record, if it has one. ProdBuild compiles every call to nothing.
  template <class F>
  void on_trace(F&& f) {
    if constexpr (Build::kMetrics) {
      if (c_.trace != nullptr) f(*c_.trace);  // on_trace: the one null test of the record
    }
  }

  // A trace stamp for this read (0 without a record). ProdBuild's is the constant 0: no branch and no clock.
  int64_t trace_stamp() {
    if constexpr (Build::kMetrics) {
      return stamp(c_.trace);  // trace_stamp: the one stamp of the record
    } else {
      return 0;
    }
  }

  // A diagnostic counter's value: 0 in ProdBuild, which keeps none.
  int64_t metric(int64_t ReaderMetrics::* field) const {
    if constexpr (Build::kMetrics) {
      return metrics_.*field;
    } else {
      (void)field;
      return 0;
    }
  }

  // Adds to a diagnostic counter; ProdBuild compiles it to nothing.
  void add_metric(int64_t ReaderMetrics::* field, int64_t n = 1) {
    if constexpr (Build::kMetrics) {
      metrics_.*field += n;
    } else {
      (void)field;
      (void)n;
    }
  }

  // No completion is withheld (the hold_ordinal fault): always true in ProdBuild.
  bool held_empty() const {
    if constexpr (Build::kFaults) {
      return faults_.held.empty();
    } else {
      return true;
    }
  }

  // The ring's depth, which is also its credit: how many SQEs may be outstanding at once. Credit-based preparation in
  // refill() means this bounds concurrency, not batch size: a batch larger than the ring waits for credit rather than
  // overrunning it. The default scales with the mirror parts, so splitting a row across roots does not halve the rows
  // in flight, and with the legs per read when reads are cut (credit counts SQEs), so the rows in flight stay what
  // they were uncut. SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH overrides it.
  unsigned queue_depth() const {
    if (configured_queue_depth_ != 0) return configured_queue_depth_;
    const size_t per_read = cuts_ ? leg_stride_ : 1;
    return static_cast<unsigned>(
        std::min<size_t>(32768, static_cast<size_t>(kQueueDepth) * static_cast<size_t>(t_.parts) * per_read));
  }

  // The next descriptor generation. 0 is reserved for "retired", so the counter skips it when it wraps.
  uint32_t next_generation() {
    if (++generation_ == 0) {
      ++generation_;
      add_metric(&ReaderMetrics::generation_wraps);
    }
    return generation_;
  }

  // Sizes the per-file limits (read cuts) and the leg and iovec strides every descriptor's storage uses. With cuts off
  // a read has one leg, or (fixed modes) one per registered buffer its iovecs meet: at most max_iovecs(). Throws when
  // the cut would need more than kMaxLegs legs.
  void size_legs() {
    cuts_ = cuts_requested_ || leg_cut_cap_ > 0;
    limits_.clear();
    for (size_t f = 0; cuts_ && f < fds_.size(); ++f) {  // sysfs is read only when reads are cut
      if (leg_cut_cap_ > 0) {
        limits_.push_back(
            DeviceLimits{std::max(kCutPage, leg_cut_cap_ / kCutPage * kCutPage), kFallbackVirtMask, "test cap"});
      } else {
        limits_.push_back(device_limits(fds_[f]));
      }
    }
    // io_ is not initialized yet, so the fixed read mode comes from the options (fixed_reads() reads io_ after init).
    bool fixed = false;
    if constexpr (requires(const Reader& reader, const iovec* v, unsigned c, FixedLeg* out) {
                    reader.fixed_legs(v, c, out);
                  }) {
      fixed = fixed_requested_;
    }
    const size_t iovecs = std::max<size_t>(1, derived().max_iovecs());
    const size_t want = cuts_ ? leg_bound(longest_read(), iovecs, min_cut_bytes()) : fixed ? iovecs : 1;
    if (want > kMaxLegs) {
      throw std::runtime_error(
          error_prefix<Layout>() + "reads cut at " + std::to_string(min_cut_bytes()) + " B need up to " +
          std::to_string(want) + " legs, more than " + std::to_string(kMaxLegs) +
          "; the cut is too small for reads of " + std::to_string(longest_read()) + " B");
    }
    leg_stride_ = static_cast<unsigned>(want);
    iov_stride_ = iovecs + (cuts_ ? leg_stride_ : 0);
  }

  // The length of the longest extent in the table.
  int64_t longest_read() const {
    int64_t longest = 0;
    for (const auto& e : t_.extents)
      longest = std::max(longest, e.length);
    return longest;
  }

  // Sizes every buffer the pipeline uses: a descriptor per (bounce slot, part, sub-read), a queue that can hold each
  // descriptor once (an extent waits in it at most once at a time), and completion and resubmission lists bounded by
  // the same count. With piece streaming off there is one sub-read per part, so this is a descriptor per (bounce slot,
  // part) and nothing piece-related is allocated. Returns false when the count does not fit the tag's index field.
  bool size_extents() {
    const size_t extents = static_cast<size_t>(kBounceSlots) * static_cast<size_t>(t_.parts) * subs_;
    // A completion carries its descriptor index in the low 24 bits of user_data (make_tag). process() rejects an index
    // past descs_.size(), but a count that does not fit in 24 bits would truncate on the way OUT, so a completion would
    // name a different live descriptor and pass that check: bytes would be credited to the wrong extent.
    if (extents > 0xFFFFFFull) return false;
    descs_.assign(extents, ExtentDesc{});
    legs_.assign(extents * leg_stride_, Leg{});
    queue_.assign(extents, 0);
    completions_.reserve(extents + 1);
    if constexpr (Build::kFaults) faults_.held.reserve(extents);
    again_.reserve(extents);
    sub_reads_.assign(piece_stream_ ? extents : 0, Read{});
    // Each descriptor's iovecs (max_iovecs), built when its legs are planned (destination). A read lies inside the
    // row image and the segments tile it, so it touches at most one run per segment (image_iovecs).
    iovecs_.assign(extents * iov_stride_, iovec{});
    iov_scratch_.assign(iov_stride_, iovec{});
    cut_scratch_.assign(leg_stride_, CutLeg{});
    fixed_scratch_.assign(iov_stride_, FixedLeg{});
    piece_runs_.assign(
        piece_stream_ ? static_cast<size_t>(kBounceSlots * kPieces) * t_.segments.size() : 0, PieceRun{});
    return true;
  }

  // Starts a read() with every descriptor retired and every bank free. This is also what makes a failed call safe to
  // follow: drain() has already retired the kernel's side of everything.
  void reset_pipeline() {
    release_drive_share();
    for (auto& d : descs_)
      d = ExtentDesc{};
    for (auto& r : rows_)
      r = BounceRow{};
    for (int b = 0; b < kBanks; ++b)
      rows_busy_[b] = bank_live_[b] = 0;
  }

  // Whether there is a row to pack; with piece streaming, a vetted piece not yet handed to a job, whatever its row's
  // state.
  bool has_ready() const {
    for (const auto& r : rows_) {
      if (piece_stream_ ? (r.vetted & static_cast<uint8_t>(~r.dispatched)) != 0 : r.state == RowState::Ready)
        return true;
    }
    return false;
  }

  // Appends descriptor `index` to the credit queue. Throws if the queue is full.
  void queue_push(uint32_t index) {
    Call& c = c_;
    // queue_ holds each descriptor at most once, so queue_count + pending <= descs_.size() == queue_.size(): a
    // descriptor leaves the queue before it is prepared and only re-enters (through again_) after its completion was
    // reaped. If that ever broke, the modulo below would overwrite the queue's head and silently drop an extent's read
    // while its row still packed and published. The check is cheap, and there is no safe way to continue.
    if (c.queue_count >= queue_.size()) {
      throw std::runtime_error(error_prefix<Layout>() + "the extent queue overflowed its descriptor count");
    }
    queue_[(c.queue_head + c.queue_count) % queue_.size()] = index;
    ++c.queue_count;
    descs_[index].queued = true;
  }

  // Queue descriptor `index` again (through again_) unless it already is: however many of its legs need resubmitting,
  // it waits in the queue once, and refill() prepares all its Idle legs together.
  void requeue(uint32_t index) {
    ExtentDesc& d = descs_[index];
    if (d.queued) return;
    d.queued = true;
    again_.push_back(index);
  }

  // A completion's tag, the SQE's user_data: generation << 32 | leg << 24 | descriptor index (index < 2^24, checked by
  // size_extents).
  static uint64_t make_tag(uint32_t generation, uint32_t index, unsigned leg) {
    return (static_cast<uint64_t>(generation) << 32) | (static_cast<uint64_t>(leg) << 24) | index;
  }
  static uint32_t tag_index(uint64_t tag) {
    return static_cast<uint32_t>(tag & 0xFFFFFFu);
  }
  static unsigned tag_leg(uint64_t tag) {
    return static_cast<unsigned>(tag >> 24 & 0xFFu);
  }
  static uint32_t tag_generation(uint64_t tag) {
    return static_cast<uint32_t>(tag >> 32);
  }

  // Whether the ring reads into registered buffers (a fixed read mode).
  bool fixed_reads() const {
    if constexpr (requires(const Reader& reader) { reader.fixed_reads(); }) {
      return io_.fixed_reads();
    } else {
      return false;
    }
  }

  // Plans a descriptor's legs on its first preparation: its iovecs from `done` 0 (destination); with read cuts,
  // rewritten into runs within its file's device limits (cut_legs); in a fixed read mode each run cut again at
  // registered-buffer changes (fixed_legs). Without either it is one leg covering the read. Each leg's `start` is the
  // prefix sum of the earlier legs' bytes; a leg wholly past the read's end-of-file expectation is Done at once.
  void plan_legs(uint32_t index) {
    ExtentDesc& d = descs_[index];
    iovec* iov = &iovecs_[static_cast<size_t>(index) * iov_stride_];
    unsigned count = derived().destination(d, iov);
    Leg* legs = &legs_[static_cast<size_t>(index) * leg_stride_];
    CutLeg* runs = cut_scratch_.data();
    unsigned run_count = 1;
    runs[0] = CutLeg{0, count, d.read->length, false};
    if (cuts_) {
      run_count = cut_legs(
          iov,
          count,
          limits_[d.read->file],
          iov_scratch_.data(),
          static_cast<unsigned>(iov_stride_),
          runs,
          leg_stride_);
      if (run_count > leg_stride_)
        throw std::logic_error(error_prefix<Layout>() + "a read needs more legs than open() sized");
      count = runs[run_count - 1].first + runs[run_count - 1].count;
      std::copy(iov_scratch_.begin(), iov_scratch_.begin() + count, iov);
      if constexpr (Build::kMetrics) {
        if (run_count > 1) ++metrics_.cut_reads;
        for (unsigned r = 0; r < run_count; ++r)
          metrics_.gap_cuts += runs[r].gap ? 1 : 0;
      }
    }
    unsigned n = 0;
    int64_t start = 0;
    for (unsigned r = 0; r < run_count; ++r) {
      FixedLeg* parts = fixed_scratch_.data();
      unsigned k = 1;
      parts[0] = FixedLeg{0, runs[r].count, -1, static_cast<size_t>(runs[r].bytes)};
      if constexpr (requires(const Reader& reader, const iovec* v, unsigned c, FixedLeg* out) {
                      reader.fixed_legs(v, c, out);
                    }) {
        if (fixed_reads()) k = io_.fixed_legs(iov + runs[r].first, runs[r].count, parts);
      }
      for (unsigned l = 0; l < k; ++l) {
        if (n == leg_stride_)
          throw std::logic_error(error_prefix<Layout>() + "a read needs more legs than open() sized");
        const int64_t bytes = static_cast<int64_t>(parts[l].bytes);
        legs[n++] =
            Leg{start,
                bytes,
                std::clamp<int64_t>(d.expected - start, 0, bytes),
                0,
                runs[r].first + parts[l].first,
                parts[l].count,
                parts[l].buffer,
                LegState::Idle};
        start += bytes;
      }
    }
    if (start != d.read->length) throw std::logic_error(error_prefix<Layout>() + "a read's legs do not cover the read");
    if constexpr (requires(Reader& reader) { reader.note_fanout(1u); }) {
      if (fixed_reads()) io_.note_fanout(n);
    }
    if constexpr (Build::kMetrics) {
      if (fixed_reads() && n > 1) {
        ++metrics_.fixed_cuts;
        metrics_.fanout_sqes += n;
      }
    }
    for (unsigned l = 0; l < n; ++l)
      if (legs[l].expected == 0) legs[l].state = LegState::Done;
    d.legs = static_cast<uint8_t>(n);
  }

  // Drops from the front of leg `g`'s iovec slice what already landed (g.done), in place. The kernel is not holding
  // these iovecs (the leg was reaped), and the slice then describes exactly the leg's remaining bytes.
  void advance_leg(iovec* iov, Leg& g) {
    size_t have = 0;
    for (unsigned i = 0; i < g.iov_count; ++i)
      have += iov[g.first_iov + i].iov_len;
    size_t skip = have - static_cast<size_t>(g.bytes - g.done);
    while (skip > 0 && skip >= iov[g.first_iov].iov_len) {
      skip -= iov[g.first_iov].iov_len;
      ++g.first_iov;
      --g.iov_count;
    }
    if (skip > 0) {
      iov[g.first_iov].iov_base = static_cast<uint8_t*>(iov[g.first_iov].iov_base) + skip;
      iov[g.first_iov].iov_len -= skip;
    }
  }

  // Admits batches while a bank is free. The abandon check comes first: once it says stop, no further work is
  // submitted, but what was already submitted is reaped by the loop.
  template <class Abandon>
  void admit(Abandon& abandon) {
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
        if (!c.stalled) on_trace([&](StageRecord& t) { ++t.bank_stalls; });
        c.stalled = true;
        return;
      }
      // The latch clears only once this turn really admits: returning below for the reading-rows cap leaves the same
      // busy bank to re-arm the edge next turn and count the SAME wait again. With step 1 and max_reading 1 that
      // interleaving is the normal case, so one wait spanning three turns would be reported as three stalls.
      if (c.reading_rows != 0 && c.reading_rows + count > c.max_reading) return;
      c.stalled = false;
      if (!admit_batch(bank, first, count)) {
        c.failed = true;
        return;
      }
      ++c.next_batch;
    }
  }

  // Validates rows [first, first + count) of the call, then takes `bank` for them: marks their slots Reading and queues
  // their extents (or sub-reads). Returns false, with nothing changed, if the bank is not free or a row cannot be read
  // whole (see the checks below).
  bool admit_batch(size_t bank, size_t first, size_t count) {
    Call& c = c_;
    const size_t parts = static_cast<size_t>(t_.parts);
    if (bank_live_[bank] != 0) return false;  // an extent still names this bank: never reuse it
    // Nor may a row still be packing (a worker holds its copy) or ready in one of its slots: rows_busy_ says so, and
    // this refuses if the two ever disagree instead of overwriting a row a worker is reading.
    for (size_t i = 0; i < count; ++i) {
      if (rows_[bank * kBounceRows + i].state != RowState::Free) return false;
    }
    // Validate the whole batch before touching any state, so a bad row leaves nothing to undo.
    for (size_t i = 0; i < count; ++i) {
      const size_t row_index = static_cast<size_t>(c.layer * t_.experts + c.experts[first + i]);
      const size_t base = row_index * parts;
      // The bytes the expert needs are [start, start + need_end) of its aligned superset and must all lie inside the
      // file. An extent's page-aligned tail may overrun end of file: that is padding no one needs, so the expectation
      // below is shortened for it. A row that needs bytes the file does not have is corrupt and must fail, not publish
      // stale bytes.
      // The head is the row's FIRST READING part, not part 0. A root whose split weight is 0 gives a zero-length part 0
      // (SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=0:1 is a supported setting), and the eager reader (exl3_row_reader.py) skips
      // zero-length parts before computing offsets, so a table builder may leave them zeroed. Reading the base and
      // the file size from such an extent would silently aim this check at the wrong file; every part the head can be
      // is one the reader submits.
      const Read* head = nullptr;
      for (size_t p = 0; p < parts && head == nullptr; ++p) {
        if (t_.extents[base + p].length > 0) head = &t_.extents[base + p];
      }
      // A row with no extent reads nothing, so packing it would publish stale bytes.
      if (head == nullptr) return false;
      if (head->offset - head->dest + t_.starts[row_index] + t_.need_end > t_.file_sizes[head->file]) return false;
      // Every reading part must have at least one byte in its own file. A builder table never violates this (a part
      // spans whole pages and a superset overruns end of file by less than one page, so a reading part always starts
      // strictly inside the file), which is why it must fail loudly here instead of being clamped to an expectation
      // of zero below: an extent that expects nothing retires having read nothing while its row still packs and
      // publishes, the corruption mode this reader exists to prevent.
      for (size_t p = 0; p < parts; ++p) {
        const Read& e = t_.extents[base + p];
        if (e.length > 0 && t_.file_sizes[e.file] - e.offset <= 0) return false;
      }
      // The slot is Free (checked above), so its piece runs may be written before the batch is known good.
      if (piece_stream_ && !plan_pieces(bank * kBounceRows + i, row_index, geometry_[i])) return false;
    }
    const int64_t admitted = trace_stamp();
    on_trace([&](StageRecord& t) { ++t.batches; });
    for (size_t i = 0; i < count; ++i) {
      const size_t ordinal = first + i;
      on_trace([&](StageRecord& t) {
        if (ordinal < static_cast<size_t>(kTraceRows)) t.row_admit[ordinal] = admitted;
      });
      const size_t slot = bank * kBounceRows + i;
      const size_t row_index = static_cast<size_t>(c.layer * t_.experts + c.experts[ordinal]);
      const size_t base = row_index * parts;
      rows_[slot] = BounceRow{RowState::Reading, ordinal, 0};
      rows_[slot].needed = t_.starts[row_index] + t_.need_end;
      ++rows_busy_[bank];
      ++c.reading_rows;
      if constexpr (Build::kFaults) {
        if (faults_.fault.poison) derived().poison_slot(slot, kPoisonFill);
      }
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
        // Per extent, against the file that extent reads. At least 1 byte: the validation loop above refused the
        // batch if any reading extent started at or past end of file.
        d.expected = std::min(extent->length, t_.file_sizes[extent->file] - extent->offset);
        d.generation = next_generation();
        d.slot = static_cast<int32_t>(slot);
        arm_stale(index);
        ++rows_[slot].extents_left;
        ++bank_live_[bank];
        queue_push(index);
        on_trace([&](StageRecord& t) {
          const size_t drive = file_drive_[extent->file];
          t.drive_dev[drive] = drive_dev_[drive];
          t.drive_extents[drive] += 1;
          const int64_t trace_slot = t.extents++;
          if (trace_slot < kTraceExtents) {
            t.extent_id[trace_slot] = (static_cast<int64_t>(ordinal) << 16) | static_cast<int64_t>(p);
            d.trace_slot = static_cast<int32_t>(trace_slot);
          } else {
            ++t.extents_untraced;
          }
        });
      }
    }
    on_trace([&](StageRecord& t) {
      t.rows_reading_max = std::max<int64_t>(t.rows_reading_max, static_cast<int64_t>(c.reading_rows));
    });
    return true;
  }

  // Piece streaming: cuts the row into sub-reads and pieces, into `g` and slot `slot`'s piece runs. Returns false to
  // refuse the batch: the row cannot be cut, or a sub-read would start at or past end of file (the per-part check in
  // admit_batch, per sub-read).
  bool plan_pieces(size_t slot, size_t row_index, RowGeometry& g) {
    const size_t segments = t_.segments.size();
    if (!row_geometry(t_, row_index, g, &piece_runs_[slot * kPieces * segments])) return false;
    for (int s = 0; s < g.subs; ++s) {
      if (t_.file_sizes[g.sub[s].file] - g.sub[s].offset <= 0) return false;
    }
    return true;
  }

  // Piece streaming: queues the row's sub-reads, one descriptor each, (slot, part, k) -> (slot * parts + part) *
  // kSubReads + k. The stride is the most sub-reads a part can have, and k < sub_reads_per_part <= kSubReads, so the
  // index is unique whatever the row's cut. Credit is untouched: refill() takes it per SQE, so a sub-read costs one
  // like a part did. Pieces with no bytes (past the row's sub-reads) are vetted here, at admission.
  void queue_sub_reads(size_t slot, size_t ordinal, const RowGeometry& g, int64_t admitted) {
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
      // Clamped at end of file per sub-read; at least 1 byte (plan_pieces).
      d.expected = std::min(extent->length, t_.file_sizes[extent->file] - extent->offset);
      d.generation = next_generation();
      d.slot = static_cast<int32_t>(slot);
      d.sub = s;
      arm_stale(index);
      r.sub_dest[s] = extent->dest;
      ++r.extents_left;
      ++bank_live_[bank];
      queue_push(index);
      on_trace([&](StageRecord& t) {
        // With the dynamic root choice the file is decided later, at first preparation, and counted there.
        if (!dynamic_) count_drive_extent(t, extent->file);
        const int64_t trace_slot = t.extents++;
        if (trace_slot < kTraceExtents) {
          t.extent_id[trace_slot] = (static_cast<int64_t>(ordinal) << 16) | (static_cast<int64_t>(g.k[s]) << 8) |
                                    static_cast<int64_t>(g.part[s]);
          d.trace_slot = static_cast<int32_t>(trace_slot);
        } else {
          ++t.extents_untraced;
        }
      });
    }
    vet_pieces(slot, admitted);
  }

  // Piece streaming: sub-read `sub` of the row in `slot` retired with `done` bytes. Vets every piece it completes.
  void land_sub_read(size_t slot, int32_t sub, int64_t done, int64_t returned) {
    Call& c = c_;
    BounceRow& r = rows_[slot];
    r.landed |= static_cast<uint8_t>(1u << sub);
    r.sub_done[sub] = done;
    const int64_t seq = ++c.events;
    on_trace([&](StageRecord& t) {
      if (r.ordinal < static_cast<size_t>(kTraceRows)) t.sub_land_seq[r.ordinal][sub] = seq;
    });
    vet_pieces(slot, returned);
  }

  // Vets each piece of `slot` whose dependencies have all landed and that is not vetted yet. A piece whose bytes the
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
      on_trace([&](StageRecord& t) {
        ++t.pieces_vetted;
        if (r.ordinal < static_cast<size_t>(kTraceRows)) {
          t.piece_cqe[r.ordinal][j] = when;
          t.piece_seq[r.ordinal][j] = seq;
        }
      });
    }
  }

  // Whether every byte of piece j lies inside a landed sub-read's [dest, dest + done). Sub-reads are in dest order, so
  // one pass per run suffices. This replaces the whole-row `filled >= needed` check: a sub-read short at end of file
  // covers only what it returned.
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

  // Prepares as many queued extents as credit and SQ room allow. `pending` counts SQEs prepared and not yet reaped (in
  // the SQ ring or in the kernel) and never exceeds `capacity`. Credit counts SQEs, not rows or logical reads, because
  // rows do not all issue the same number of reads: a root serving none of a row issues nothing, and a fanned-out
  // fixed read issues one SQE per leg. A descriptor's Idle legs are reserved all-or-nothing: prepared together only if
  // they fit the credit (or nothing is pending, so a read wider than a lowered credit still progresses) and the SQ;
  // otherwise it stays at the queue head. Retries re-enter through the queue, so they take credit like any other read.
  //
  // A gated root policy (IdleRoots) also decides, before a sub-read's first preparation, whether it may go now and from
  // which root (gate_root); a sub-read it holds back stays at the queue head unplanned and is decided again next turn.
  template <class RootPolicy>
  void refill(RootPolicy& roots) {
    Call& c = c_;
    int64_t prepared = 0;  // one clock read per refill turn, taken on the first SQE
    clock_ = 0;            // and at most one for the drive load (DriveLoad::change)
    while (c.queue_count > 0) {
      const uint32_t index = queue_[c.queue_head];
      ExtentDesc& d = descs_[index];
      if constexpr (std::decay_t<RootPolicy>::kGated) {
        if (d.legs == 0 && !gate_root(index, roots)) break;
      }
      if (d.legs == 0) {
        // First preparation: with the dynamic root choice, the sub-read's root is picked now, from the load as it is
        // when the read is issued, before its legs are cut by that root's device limits. A gated read's root is
        // gate_root's.
        if constexpr (!std::decay_t<RootPolicy>::kGated) {
          if (dynamic_) choose_root(index);
        }
        plan_legs(index);
      }
      // First attempt: nothing landed and nothing retried (a descriptor is re-queued only by a short or retried leg).
      const bool fresh = d.done == 0 && d.retries == 0;
      Leg* legs = &legs_[static_cast<size_t>(index) * leg_stride_];
      unsigned n = 0;
      for (unsigned l = 0; l < d.legs; ++l)
        n += legs[l].state == LegState::Idle ? 1u : 0u;
      bool room = c.pending + n <= c.capacity || c.pending == 0;
      if constexpr (requires(const Reader& reader) { reader.sq_space(); }) {
        room = room && !(n > 1 && io_.sq_space() < n);
      }
      if (!room) {
        // Not issued: a fresh sub-read under the dynamic choice chooses again when credit lets it go, so the choice
        // sees the load at issue, not the load of the turn it first waited at the queue head.
        if (dynamic_ && fresh) d.legs = 0;
        break;
      }
      // Issued: the dynamic choice is final, so it is counted now (once: only a fresh sub-read is issued first).
      if (dynamic_ && fresh) note_root(index);
      c.queue_head = (c.queue_head + 1) % queue_.size();
      --c.queue_count;
      d.queued = false;
      // A resubmission (short read, -EINTR/-EAGAIN) starts past what already landed: advance_leg trims the leg's
      // iovecs, and a mid-file O_DIRECT short read ends on a block boundary, so the remainder is a legal direct read.
      iovec* iov = &iovecs_[static_cast<size_t>(index) * iov_stride_];
      const int fd = fds_[d.read->file];
      for (unsigned l = 0; l < d.legs; ++l) {
        Leg& g = legs[l];
        if (g.state != LegState::Idle) continue;
        if (g.done > 0) advance_leg(iov, g);
        const int64_t remaining = g.bytes - g.done;
        const uint64_t tag = make_tag(d.generation, index, l);
        const uint64_t offset = static_cast<uint64_t>(d.read->offset + g.start + g.done);
        const iovec* slice = &iov[g.first_iov];
        bool prepared_one;
        if constexpr (requires(Reader& reader, const iovec* v, uint64_t at) {
                        reader.prep_readv_fixed(0, v, 1u, at, 0, at);
                      }) {
          if (fixed_reads()) {
            prepared_one = io_.prep_readv_fixed(fd, slice, g.iov_count, offset, g.buffer, tag);
          } else {
            prepared_one = prep_default(fd, slice, g.iov_count, offset, tag);
          }
        } else {
          prepared_one = prep_default(fd, slice, g.iov_count, offset, tag);
        }
        // Credit and SQ room were checked for every Idle leg above.
        if (!prepared_one) throw std::logic_error(error_prefix<Layout>() + "an SQE was refused after it was reserved");
        g.state = LegState::Inflight;
        ++d.legs_inflight;
        ++c.pending;
        count_drive(d, d.legs_inflight == 1 ? 1 : 0, remaining);  // the sub-read's first leg in flight counts the read
        if constexpr (Build::kFaults) {
          if (faults_.sqe_log) {
            faults_.sqe_log->push_back(
                SqeRecord{
                    d.read->file,
                    d.read->offset + g.start + g.done,
                    remaining,
                    static_cast<int64_t>(d.slot) * t_.slot_bytes + d.read->dest + g.start + g.done});
          }
        }
        on_trace([&](StageRecord& t) {
          t.submitted_bytes += remaining;
          if (g.done > 0 || d.retries > 0) t.retried_bytes += remaining;
          t.pending_max = std::max<int64_t>(t.pending_max, static_cast<int64_t>(c.pending));
          if (d.trace_slot >= 0) {
            if (fresh) {
              if (t.extent_submit[d.trace_slot] == 0) {
                if (prepared == 0) prepared = trace_stamp();
                t.extent_submit[d.trace_slot] = prepared;
              }
            } else {
              ++t.extent_attempts[d.trace_slot];
            }
          }
        });
      }
      if constexpr (std::decay_t<RootPolicy>::kGated) {
        if (fresh && roots.on_piece != nullptr)
          roots.on_piece(roots.context, DriveLoad::root_of(d.read->file, t_.parts));
      }
    }
  }

  // Prepares an SQE with the default opcode: IORING_OP_READV for a scattering reader (Derived::kScatter), else
  // IORING_OP_READ (one leg, one iovec).
  bool prep_default(int fd, const iovec* iov, unsigned count, uint64_t offset, uint64_t tag) {
    return Derived::kScatter ? io_.prep_readv(fd, iov, count, offset, tag)
                             : io_.prep_read(fd, iov[0].iov_base, static_cast<unsigned>(iov[0].iov_len), offset, tag);
  }

  // Submits, waits for a completion only when `ready` is false, then drains the CQ before processing it, so the CQ
  // frees early and a fault can reorder the completions. The fault branches exist in InstrBuild only.
  void reap(bool ready) {
    Call& c = c_;
    if constexpr (Build::kFaults) {
      if (!ready && c.pending == 0 && !faults_.held.empty()) {
        release_held();
        return;
      }
    }
    on_trace([&](StageRecord& t) {
      if (c.submitted == 0) c.submitted = trace_stamp();
    });
    unsigned wait_nr = ready ? 0u : 1u;
    if constexpr (Build::kFaults) {
      // reverse_cqes reorders only one reaped batch, so wait for every read in flight; otherwise whether anything is
      // reversed depends on how the device happened to batch its completions.
      if (faults_.fault.reverse_cqes && c.pending > 0) wait_nr = c.pending;
    }
    const int rc = submit(wait_nr);
    if (rc < 0) {
      // -EINTR/-EAGAIN/-EBUSY: reap what has completed and submit again (as uring_file_reader.cpp does). Anything
      // else, or a soft error that never clears, fails.
      const bool soft = rc == -EINTR || rc == -EAGAIN || rc == -EBUSY;
      if (!soft || ++c.soft_errors > kMaxSoftErrors) {
        c.failed = true;
        return;
      }
    } else {
      c.soft_errors = 0;
    }
    const int64_t returned = trace_stamp();
    clock_ = 0;  // the drive load's clock for this reap: read at most once (DriveLoad::change)
    completions_.clear();
    again_.clear();
    const unsigned seen = io_.reap(completions_);
    c.pending -= seen;
    if constexpr (Build::kFaults) apply_reap_faults();
    for (size_t k = 0; k < completions_.size(); ++k)
      process(completions_[k], returned);
    if (seen > 0) {
      on_trace([&](StageRecord&) {
        if (c.first_seen == 0) c.first_seen = returned;
        c.last_seen = returned;
      });
    }
    if (c.failed) return;
    for (uint32_t index : again_)
      queue_push(index);
  }

  // Fault (hold_ordinal), InstrBuild only: every other row is done, so the withheld completions arrive now.
  void release_held()
    requires(Build::kFaults)
  {
    Call& c = c_;
    completions_.assign(faults_.held.begin(), faults_.held.end());
    faults_.held.clear();
    again_.clear();
    clock_ = 0;
    const int64_t released = trace_stamp();
    for (size_t k = 0; k < completions_.size(); ++k)
      process(completions_[k], released);
    on_trace([&](StageRecord&) {
      if (c.first_seen == 0) c.first_seen = released;
      c.last_seen = released;
    });
    if (!c.failed) {
      for (uint32_t index : again_)
        queue_push(index);
    }
  }

  // Applies the reap faults to one reaped batch before it is processed: reverse_cqes, hold_ordinal and the stale
  // redelivery. InstrBuild only.
  void apply_reap_faults()
    requires(Build::kFaults)
  {
    const ReadFault& fault = faults_.fault;
    if (fault.reverse_cqes) std::reverse(completions_.begin(), completions_.end());
    if (fault.hold_ordinal >= 0) {
      size_t kept = 0;
      for (size_t k = 0; k < completions_.size(); ++k) {
        const uint32_t index = tag_index(completions_[k].data);
        const bool live = index < descs_.size() && descs_[index].generation != 0 &&
                          descs_[index].generation == tag_generation(completions_[k].data);
        if (live &&
            (fault.hold_rest ? static_cast<int64_t>(rows_[descs_[index].slot].ordinal) >= fault.hold_ordinal
                             : static_cast<int64_t>(rows_[descs_[index].slot].ordinal) == fault.hold_ordinal) &&
            (fault.sub < 0 || fault_matches_sub(index)) &&
            (fault.leg < 0 || static_cast<int64_t>(tag_leg(completions_[k].data)) == fault.leg)) {
          faults_.held.push_back(completions_[k]);
        } else {
          completions_[kept++] = completions_[k];
        }
      }
      completions_.resize(kept);
    }
    // A completion of an extent that retired earlier arrives after its descriptor was recycled. It names a dead
    // generation, so it must fail the read and touch nothing; without the generation it would complete whichever
    // extent now lives in that descriptor, publishing bytes never read.
    if (faults_.stale_armed) {
      completions_.push_back(faults_.stale);
      faults_.stale_armed = faults_.stale_waiting = false;
    }
  }

  // Fault (stale_cqe_call): descriptor `index` was just recycled; if the held completion named it, deliver it with
  // the next reap. Does nothing in ProdBuild.
  void arm_stale(uint32_t index) {
    if constexpr (Build::kFaults) {
      if (faults_.stale_waiting && index == faults_.stale_index) faults_.stale_armed = true;
    } else {
      (void)index;
    }
  }

  // Applies one completion to its leg: retries a transient error, fails the read on a hard one or a stale tag, and
  // retires the extent when its last leg is done. A short read resubmits just the rest of that leg.
  void process(const Completion& completion, int64_t returned) {
    Call& c = c_;
    const uint32_t index = tag_index(completion.data);
    const unsigned l = tag_leg(completion.data);
    const uint32_t generation = tag_generation(completion.data);
    // Stale unless it names a live descriptor's leg that is in flight: a dead generation, a leg the read does not
    // have, or a leg already reaped.
    if (index >= descs_.size() || generation == 0 || descs_[index].generation != generation ||
        l >= descs_[index].legs || legs_[static_cast<size_t>(index) * leg_stride_ + l].state != LegState::Inflight) {
      add_metric(&ReaderMetrics::stale_cqes);
      c.failed = true;
      return;
    }
    ExtentDesc& d = descs_[index];
    Leg& g = legs_[static_cast<size_t>(index) * leg_stride_ + l];
    g.state = LegState::Idle;
    --d.legs_inflight;
    // The leg's bytes leave flight whatever it returned (done is unchanged since it was prepared), and the sub-read's
    // last leg reaped takes its read back; a retry or a short leg counts again when it is prepared again.
    count_drive(d, d.legs_inflight == 0 ? -1 : 0, -(g.bytes - g.done));
    int res = completion.res;
    bool eof = false;  // fault (short_is_eof): this completion ends the sub-read
    add_metric(&ReaderMetrics::cqes);
    if constexpr (Build::kFaults) {
      const ReadFault& fault = faults_.fault;
      // A fault's part is the descriptor's part whatever the sub-read count (subs_ is 1 with the flag off).
      const size_t part = (index / subs_) % static_cast<size_t>(t_.parts);
      const bool leg_matches = fault.leg < 0 || static_cast<int64_t>(l) == fault.leg;
      if (fault.cqe_error != 0 && metric(&ReaderMetrics::cqes) == fault.cqe_call && leg_matches) res = -fault.cqe_error;
      const bool part_matches = fault.part < 0 || static_cast<int64_t>(part) == fault.part;
      const bool root_matches = fault.root < 0 || DriveLoad::root_of(d.read->file, t_.parts) == fault.root;
      if ((fault.part >= 0 || fault.root >= 0) && !faults_.part_fired && part_matches && root_matches && leg_matches &&
          (fault.sub < 0 || static_cast<int64_t>(index % subs_) == fault.sub) &&
          (fault.ordinal < 0 || static_cast<int64_t>(rows_[d.slot].ordinal) == fault.ordinal)) {
        if (fault.part_error != 0) {
          faults_.part_fired = true;
          res = -fault.part_error;
        } else if (fault.part_short > 0 && res > fault.part_short) {
          faults_.part_fired = true;
          res = static_cast<int>(fault.part_short);
          eof = fault.short_is_eof;
        }
      }
    }
    // Two parts of one row, and two legs of one read, complete independently, so a retry resubmits one leg.
    if (res == -EINTR || res == -EAGAIN) {
      if (++d.retries > kMaxRetries) {  // the retry budget is the descriptor's
        c.failed = true;
      } else {
        requeue(index);  // resubmit the same range
      }
      return;
    }
    if (res < 0 || (res == 0 && g.done < g.expected)) {
      c.failed = true;
      return;
    }
    g.done += res;
    d.done += res;
    load_->landed(DriveLoad::root_of(d.read->file, t_.parts), kind_, res);
    if (eof) {
      d.expected -= g.expected - g.done;
      g.expected = g.done;
    }
    // A mid-file O_DIRECT read ends short only on a logical-block boundary, so offset + done, dest + done and
    // length - done stay block-aligned (an extent's offset, dest and length are whole pages) and the resubmit is a
    // legal direct read of just this leg. At end of file done == expected: no resubmit.
    if (g.done < g.expected) {
      requeue(index);
      return;
    }
    g.state = LegState::Done;
    if (d.legs_inflight != 0) return;
    const Leg* legs = &legs_[static_cast<size_t>(index) * leg_stride_];
    for (unsigned k = 0; k < d.legs; ++k)
      if (legs[k].state != LegState::Done) return;
    retire(index, completion, returned);
  }

  // Fault (hold_ordinal with sub): whether descriptor `index` is sub-read fault.sub of part fault.part (any part at
  // -1).
  bool fault_matches_sub(uint32_t index) const
    requires(Build::kFaults)
  {
    const ReadFault& fault = faults_.fault;
    const int64_t part = static_cast<int64_t>((index / subs_) % static_cast<size_t>(t_.parts));
    return static_cast<int64_t>(index % subs_) == fault.sub && (fault.part < 0 || part == fault.part);
  }

  // Handles an extent's last completion: accounts it, retires the descriptor, and marks the row ready to pack when it
  // was the row's last extent. Nothing reads the descriptor afterwards.
  void retire(uint32_t index, const Completion& completion, int64_t returned) {
    Call& c = c_;
    ExtentDesc& d = descs_[index];
    if (d.legs_inflight != 0) throw std::logic_error(error_prefix<Layout>() + "an extent retired with a leg in flight");
    // What the read delivered contiguously from its start: the legs in order, up to the first that ended short (only
    // end of file, or the short_is_eof fault, ends a leg short). With one leg this is d.done.
    int64_t delivered = 0;
    const Leg* legs = &legs_[static_cast<size_t>(index) * leg_stride_];
    for (unsigned k = 0; k < d.legs; ++k) {
      delivered += legs[k].done;
      if (legs[k].done < legs[k].bytes) break;
    }
    on_trace([&](StageRecord& t) {
      const size_t drive = file_drive_[d.read->file];
      t.drive_bytes[drive] += d.done;
      t.bytes += d.done;
      if (d.trace_slot >= 0) t.extent_cqe[d.trace_slot] = returned;
    });
    const size_t slot = static_cast<size_t>(d.slot);
    rows_[slot].filled += delivered;
    if (piece_stream_) land_sub_read(slot, d.sub, delivered, returned);
    --bank_live_[slot / kBounceRows];
    if constexpr (Build::kFaults) {
      if (faults_.fault.stale_cqe_call > 0 && ++faults_.retired == faults_.fault.stale_cqe_call) {
        faults_.stale = completion;
        faults_.stale_index = index;
        faults_.stale_waiting = true;
      }
    } else {
      (void)completion;
    }
    d = ExtentDesc{};
    if constexpr (Build::kFaults) {
      if (faults_.fault.poison) d.slot = kPoisonSlot;
    }
    if (--rows_[slot].extents_left == 0) {
      rows_[slot].state = RowState::Ready;
      --c.reading_rows;
    }
  }

  // The earliest row in request order among those ready, vetted for packing; kBounceSlots when there is none or the
  // vetting failed the call. Runs on the owner, before any copy, however the copy is done.
  size_t take_ready_row() {
    Call& c = c_;
    size_t best = kBounceSlots;
    for (size_t s = 0; s < static_cast<size_t>(kBounceSlots); ++s) {
      if (rows_[s].state != RowState::Ready) continue;
      if (best == static_cast<size_t>(kBounceSlots) || rows_[s].ordinal < rows_[best].ordinal) best = s;
    }
    if (best == static_cast<size_t>(kBounceSlots)) return best;
    // Defence in depth for the one failure this reader must never have: packing bytes no drive delivered.
    // admit_batch's end-of-file guard already refuses a row the file cannot satisfy, but it decides from the head
    // part's file size alone, so it is only as good as the table's row consistency (checked in tables_from). This
    // compares what the drives actually returned for THIS row against what its segments will read. It costs one
    // compare per row and, unlike the byte-split counters, is not behind the trace flag. Extents fill the slot
    // contiguously from dest 0 and only a tail extent can stop short without being resubmitted (a short read retries;
    // only the end-of-file clamp shortens an expectation), so a total of at least `needed` means the needed prefix is
    // whole. With piece streaming vet_pieces makes the same check per piece instead.
    if (rows_[best].filled < rows_[best].needed) {
      c.failed = true;
      return static_cast<size_t>(kBounceSlots);
    }
    return best;
  }

  // Piece streaming, the owner's publish: piece j of the row in `slot` is stored and fenced (its job is done, or it
  // had no bytes), so sets its bit on every readiness word naming the row. A word that refuses (another generation, or
  // the bit already set) fails the call. The bit is marked published either way: the piece was collected, and nothing
  // else may wait on it.
  void publish_collected(size_t slot, int j) {
    Call& c = c_;
    BounceRow& r = rows_[slot];
    const uint8_t bit = static_cast<uint8_t>(1u << j);
    bool twice = false;
    if constexpr (Build::kFaults) {
      twice = ++faults_.publishes == faults_.fault.publish_twice;  // fault: publish_twice is 0 when off
      if (++c.published == c.total * kPieces && faults_.fault.last_publish_delay_ns > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(faults_.fault.last_publish_delay_ns));
      }
    }
    if (c.publish != nullptr && c.publish->rows != nullptr) {
      const PieceTarget& target = c.publish->rows[r.ordinal];
      for (int w = 0; w < target.count; ++w) {
        for (int attempt = 0; attempt < (twice ? 2 : 1); ++attempt) {
          if (publish_piece(target.words[w], c.publish->generation, bit)) continue;
          add_metric(&ReaderMetrics::publish_refused);
          on_trace([&](StageRecord& t) { ++t.piece_publish_refused; });
          c.failed = true;
        }
      }
    }
    const int64_t seq = ++c.events;
    on_trace([&](StageRecord& t) {
      ++t.pieces_published;
      if ((r.published >> (j + 1)) != 0) ++t.pieces_out_of_order;  // a higher-numbered piece went first
      if (r.ordinal < static_cast<size_t>(kTraceRows)) t.piece_publish[r.ordinal][j] = seq;
    });
    r.published |= bit;
  }

  // The row is packed whole: accounts it, flags it and frees its slot. Packing is the last reference the bank held on
  // this slot, so only now may it be reused.
  void finish_row(size_t best, int64_t start, int64_t end) {
    Call& c = c_;
    const size_t ordinal = rows_[best].ordinal;
    on_trace([&](StageRecord& t) {
      if (ordinal < static_cast<size_t>(kTraceRows)) {
        t.row_pack_start[ordinal] = start;
        t.row_pack_end[ordinal] = end;
      } else {
        ++t.rows_untraced;
      }
      for (const Segment& segment : t_.segments)
        t.useful_bytes += segment.bytes;
      // Rows pack in completion order and may overlap, so the first to finish is not always the first to start.
      if (t.pack_start == 0 || start < t.pack_start) t.pack_start = start;
      t.pack_end = std::max(t.pack_end, end);
      t.pack_ns += end - start;
    });
    if (c.packed) (*c.packed)[ordinal] = 1;
    derived().after_finish(best);
    rows_[best] = BounceRow{};
    --rows_busy_[best / kBounceRows];
    // Last, with the reader's bookkeeping settled: a fill publishes this row's landing now, not after the turn's last
    // row, which would leave the landed prefix stale.
    if (c.progress != nullptr) c.progress(c.progress_closure);
  }

  // Accounts a failed call: what every still-live extent was owed but never returned is cancelled. Extents that
  // retired were accounted when they did.
  void account_unfinished() {
    on_trace([&](StageRecord& t) {
      for (const ExtentDesc& d : descs_) {
        if (d.generation == 0) continue;
        const size_t drive = file_drive_[d.read->file];
        t.drive_bytes[drive] += d.done;
        t.bytes += d.done;
        t.cancelled_bytes += std::max<int64_t>(0, d.expected - d.done);
      }
    });
  }

  // Submits the prepared SQEs, waiting for `wait_nr` completions (0: do not block).
  int submit(unsigned wait_nr) {
    return io_.submit(wait_nr);
  }

  // Empties the ring after a failure: settles every read prepared or in flight, so nothing can still write the slots
  // once the caller reuses or frees them. The count is zeroed before io_.drain: a failed ring reset throws out of it,
  // and Quiesce's drain on the way out must then see nothing pending, not the stale count, which no longer matches the
  // ring's (0) and would terminate.
  void drain(unsigned pending) {
    c_.pending = 0;
    release_drive_share();  // drained completions never reach process(): before io_.drain, which may throw
    io_.drain(pending);
  }

  // set_mirror_map's map: alt_file_[file * parts + q] = root q's file of `file`'s row. Throws, naming `prefix`, unless
  // piece streaming is on and every row's files are a full mirror set.
  void build_mirror_map(const std::string& prefix) {
    const int64_t parts = t_.parts;
    if (!piece_stream_) throw std::runtime_error(prefix + "choosing a sub-read's root needs piece streaming");
    if (parts > kMaxDrives)
      throw std::runtime_error(prefix + "the drive load counts at most " + std::to_string(kMaxDrives) + " roots");
    const int64_t files = static_cast<int64_t>(t_.paths.size());
    auto refuse = [&](int64_t file, const std::string& why) {
      throw std::runtime_error(
          prefix + "file " + t_.paths[static_cast<size_t>(file)] + " is not a full mirror set: " + why);
    };
    if (files % parts != 0) refuse(files - 1, "the file count is not a multiple of the mirror roots");
    for (size_t i = 0; i < t_.extents.size(); ++i) {
      const Read& e = t_.extents[i];
      if (e.length > 0 && e.file % parts != static_cast<int64_t>(i) % parts)
        refuse(e.file, "part " + std::to_string(static_cast<int64_t>(i) % parts) + " reads another root's file");
    }
    std::vector<int64_t> alt(static_cast<size_t>(files * parts));
    for (int64_t f = 0; f < files; ++f) {
      for (int64_t q = 0; q < parts; ++q) {
        const int64_t other = f - f % parts + q;
        if (t_.source_paths[static_cast<size_t>(other)] != t_.source_paths[static_cast<size_t>(f)])
          refuse(other, "it copies " + t_.source_paths[static_cast<size_t>(other)] + ", not " +
                            t_.source_paths[static_cast<size_t>(f)]);
        if (t_.file_sizes[static_cast<size_t>(other)] != t_.file_sizes[static_cast<size_t>(f)])
          refuse(other, "its size differs from its mirror's");
        alt[static_cast<size_t>(f * parts + q)] = other;
      }
    }
    alt_file_ = std::move(alt);
  }

  // The dynamic root choice for piece-stream descriptor `index`, at its first preparation (set_mirror_caps), and again
  // at each refill turn while it waits for credit (refill undoes an unissued plan). A root is
  // open while its sub-reads in flight, demand and speculative, over every reader of the drive load, are below its
  // cap. Among the open roots the one with the fewest bytes in flight wins; a tie goes to the sub-read's own root,
  // then to the lowest root. With every root at its cap the sub-read keeps its own root: the choice never blocks or
  // waits. Only the file changes: every root holds the row's image at the same offsets and size (set_mirror_caps
  // checked), so the offset, length, destination, expectation and pieces stay the static geometry's.
  void choose_root(uint32_t index) {
    Read& read = sub_reads_[index];
    const int64_t parts = t_.parts;
    // The sub-read's own root is its descriptor's part (it may have chosen another before, while it waited for credit).
    const int own = static_cast<int>((index / subs_) % static_cast<size_t>(parts));
    read.file = alt_file_[static_cast<size_t>(read.file * parts + own)];
    int best = -1;
    int64_t best_bytes = 0;
    for (int q = 0; q < static_cast<int>(parts); ++q) {
      if (load_->reads_in_flight(q) >= caps_[q]) continue;
      const int64_t bytes = load_->bytes_in_flight(q);
      if (best < 0 || bytes < best_bytes || (bytes == best_bytes && q == own)) {
        best = q;
        best_bytes = bytes;
      }
    }
    if (best >= 0 && best != own) read.file = alt_file_[static_cast<size_t>(read.file * parts + best)];
  }

  // IdleRoots, before sub-read `index`'s first preparation: true lets it go now, its file set to the root it reads;
  // false holds it back this turn, at the queue head and unplanned. A boosted read takes its table root, as a demand
  // would. Otherwise one piece goes in flight at a time, from the row's root while no demand reads it, else from the
  // demand-free root with the fewest bytes in flight; with none demand-free it waits, and past the deadline, or when
  // give_up says so, the read is abandoned (c_.failed, roots.abandoned). Relaxed loads of every reader's load: a
  // demand that lands between the check and the submit shares its drive with at most this one piece.
  bool gate_root(uint32_t index, IdleRoots& roots) {
    Call& c = c_;
    const int64_t parts = t_.parts;
    Read& read = sub_reads_[index];
    if (!roots.boosted && roots.boost != nullptr && roots.boost->load(std::memory_order_acquire) != 0)
      roots.boosted = true;
    if (roots.boosted) {
      const int own = static_cast<int>((index / subs_) % static_cast<size_t>(parts));
      read.file = alt_file_[static_cast<size_t>(read.file * parts + own)];
      return true;
    }
    if (c.reads_inflight > 0) return false;
    if (roots.give_up != nullptr && roots.give_up(roots.context)) return abandon_gated(roots);
    int root = c.gate_root >= 0 && load_->demand_reads(c.gate_root) == 0 ? c.gate_root : -1;
    for (int q = 0; root < 0 && q < static_cast<int>(parts); ++q) {
      if (load_->demand_reads(q) != 0) continue;
      int best = q;
      for (int other = q + 1; other < static_cast<int>(parts); ++other) {
        if (load_->demand_reads(other) == 0 && load_->bytes_in_flight(other) < load_->bytes_in_flight(best))
          best = other;
      }
      root = best;
    }
    if (root < 0) {
      const int64_t now = now_ns();
      if (c.deferred_since == 0) {
        c.deferred_since = now;
        ++roots.deferrals;
      } else if (now - c.deferred_since > roots.deadline_ns) {
        return abandon_gated(roots);
      }
      return false;
    }
    c.deferred_since = 0;
    if (c.gate_root >= 0 && root != c.gate_root) ++roots.moves;
    c.gate_root = root;
    read.file = alt_file_[static_cast<size_t>(read.file * parts + root)];
    return true;
  }

  // IdleRoots gives up: no further piece is issued, and read() drains what is in flight and returns 0.
  bool abandon_gated(IdleRoots& roots) {
    roots.abandoned = true;
    c_.failed = true;
    return false;
  }

  // The dynamic choice of descriptor `index` is final (its first issue): counts a redirect and the trace's extent.
  void note_root(uint32_t index) {
    const int64_t file = sub_reads_[index].file;
    const int own = static_cast<int>((index / subs_) % static_cast<size_t>(t_.parts));
    const int root = DriveLoad::root_of(file, t_.parts);
    if (root != own) load_->redirected(own, root);
    on_trace([&](StageRecord& t) { count_drive_extent(t, file); });
  }

  // The trace's per-drive extent count: one more extent reads `file`.
  void count_drive_extent(StageRecord& t, int64_t file) {
    const size_t drive = file_drive_[file];
    t.drive_dev[drive] = drive_dev_[drive];
    t.drive_extents[drive] += 1;
  }

  // Adds `reads` sub-reads and `bytes` bytes of descriptor `d`'s root to the shared drive load, and to this reader's
  // share of it.
  void count_drive(const ExtentDesc& d, int reads, int64_t bytes) {
    c_.reads_inflight += reads;
    const int root = DriveLoad::root_of(d.read->file, t_.parts);
    mine_[root][kind_].reads += reads;
    mine_[root][kind_].bytes += bytes;
    load_->change(root, kind_, reads, bytes, clock_);
  }

  // Takes this reader's share back out of the shared drive load and zeroes it: what a failed read left counted (its
  // drained completions never reach process()). A no-op after a read that returned normally.
  void release_drive_share() {
    clock_ = 0;
    for (int root = 0; root < kMaxDrives; ++root) {
      for (int kind = 0; kind < 2; ++kind) {
        Share& share = mine_[root][kind];
        if (share.reads == 0 && share.bytes == 0) continue;
        load_->change(root, kind, static_cast<int>(-share.reads), -share.bytes, clock_);
        share = Share{};
      }
    }
  }

  Tables t_;
  bool direct_;
  std::vector<int> fds_;
  std::vector<int64_t> devs_;        // st_dev of each distinct filesystem, in first-opened order
  std::vector<uint8_t> file_drive_;  // per file: its drive slot in a StageRecord
  int64_t drive_dev_[kMaxDrives] = {};
  // Pipeline state (see the class comment).
  Call c_;
  std::vector<ExtentDesc> descs_;
  std::vector<uint32_t> queue_;  // ring of descriptors waiting for credit; each appears at most once
  std::vector<Completion> completions_;
  std::vector<Leg> legs_;  // leg_stride_ per descriptor (size_extents)
  std::vector<uint32_t> again_;
  BounceRow rows_[kBounceSlots];
  // Test-only owner-pinning scaffold (set_owner_core; -1: no pin). `unpinned_affinity_` is the mask open() found
  // before pinning, restored by the destructor.
  int64_t owner_core_ = -1;
  bool owner_pinned_ = false;
  cpu_set_t unpinned_affinity_{};
  // The ring. A derived reader that owns memory registered with it closes it (close_io()) before freeing that memory;
  // ~ReaderCore closes it again (a no-op then) before closing the files it registered. This is safe only because
  // read() always drains the ring (quiesce()) before returning, so nothing is in flight when a reader is destroyed.
  Reader io_;
  unsigned configured_queue_depth_ = 0;
  bool cuts_requested_ = false;       // READ_CUTS resolved (UringOptions::read_cuts_on)
  bool fixed_requested_ = false;      // READ_MODE fixed/readv_fixed (the options, before io_ is initialized)
  bool cuts_ = false;                 // in force: requested, or a test cap
  int64_t leg_cut_cap_ = 0;           // test only (fault word leg_cut_cap): every file cut at this, 0: device limits
  std::vector<DeviceLimits> limits_;  // per file, t_.paths order
  unsigned leg_stride_ = 1;
  size_t iov_stride_ = 1;
  std::vector<iovec> iov_scratch_;
  std::vector<CutLeg> cut_scratch_;
  std::vector<FixedLeg> fixed_scratch_;
  // Piece streaming (set_piece_stream; off by default). subs_ is the most sub-reads per part: 1 with the flag off,
  // which makes descriptor (slot, part, sub) just (slot, part), and kSubReads with it on whatever a row's cut
  // (row_geometry cuts each reading part into sub_reads_per_part <= kSubReads). sub_reads_ holds each live sub-read's
  // Read (a descriptor points into it), piece_runs_ each slot's piece runs, geometry_ a batch's rows between
  // validation and admission. All are sized at open() or set_piece_stream(), and empty with the flag off.
  bool piece_stream_ = false;
  size_t subs_ = 1;
  std::vector<Read> sub_reads_;
  std::vector<PieceRun> piece_runs_;
  std::vector<iovec> iovecs_;  // direct mode: segments.size() per descriptor (size_extents)
  RowGeometry geometry_[kBounceRows];
  size_t rows_busy_[kBanks] = {};  // rows not yet packed, per bank: the packing references
  size_t bank_live_[kBanks] = {};  // extents not yet retired, per bank: the I/O references
  uint32_t generation_ = 0;
  // The drive load (drive_load.h): functional state, in both builds. `load_` is the tier's, or `own_load_` for a reader
  // on its own; `mine_` is this reader's share of it per root and kind, which every read() return leaves at 0; `clock_`
  // is the turn's clock for it (0 until read).
  struct Share {
    int64_t reads = 0;
    int64_t bytes = 0;
  };
  DriveLoad own_load_;
  DriveLoad* load_ = &own_load_;
  int kind_ = kDemandRead;
  int64_t clock_ = 0;
  Share mine_[kMaxDrives][2] = {};
  // The dynamic root choice (set_mirror_caps): on or off, each root's in-flight cap, and the mirror file map,
  // alt_file_[file * parts + q] = root q's file of `file`'s row.
  bool dynamic_ = false;
  int64_t caps_[kMaxDrives] = {};
  std::vector<int64_t> alt_file_;
  // The diagnostic counters (ReaderMetrics) and the test-only fault state (FaultState): see their types above.
  [[no_unique_address]] std::conditional_t<Build::kMetrics, ReaderMetrics, NoReaderMetrics> metrics_;
  [[no_unique_address]] std::conditional_t<Build::kFaults, FaultState, NoFaultState> faults_;
  // ProdBuild's metric and fault members are asserted empty types (with [[no_unique_address]] they take no storage).
  // This is the type-system half of the proof that ProdBuild carries no instrumentation; nm cannot see state whose
  // names are inlined away.
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(metrics_)>, "ProdBuild has no metrics");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(faults_)>, "ProdBuild has no faults");
};

}  // namespace expert_stream
}  // namespace sglang
