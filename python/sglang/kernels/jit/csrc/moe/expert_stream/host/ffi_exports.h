// The expert-stream host FFI surface, written once for every row layout and file reader. An instantiation file
// names a layout, a reader and a build policy (build_policy.h) and expands EXPERT_STREAM_HOST_EXPORTS; see
// exl3_ram_miss_host.cpp (ProdBuild) and exl3_ram_miss_host_instr.cpp (InstrBuild).
#pragma once

#include <sgl_kernel/tensor.h>

#include "../tensor_checks.h"
#include "build_policy.h"
#include "row_reader.h"
#include "ram_thread.h"

namespace sglang::expert_stream {

using tvm::ffi::TensorView;

/// \brief Every host export of one transport instantiation. Its function-local registries are per instantiation,
/// and each layout is its own module, so one layout's handles can never resolve in another's.
template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build>
struct HostExports {
  static_assert(BuildPolicy<Build>);
  using Source = RowReader<Layout, Reader, Build>;
  using Tier = RamTier<Source>;
  using Thread = RamThread<Tier>;

  // What the fault machinery compiles to in this build (plan Task 10): the instantiation files static_assert these,
  // so a ProdBuild module that regained a fault entry, an SQE log or the ballast fails to compile.
  static constexpr bool kReaderFaults = requires(Source& reader, const ReadFault& fault) { reader.set_fault(fault); };
  static constexpr bool kSqeLog = requires(Source& reader) { reader.set_sqe_log(nullptr); };
  static constexpr bool kBallast = requires(Tier& tier) { tier.copy_engine_ballast(0, 0, 0); };

  static std::vector<int64_t> slots_of(TensorView slots) {
    const auto* data = static_cast<const int64_t*>(slots.data_ptr());
    return std::vector<int64_t>(data, data + slots.size(0));
  }

  // The table tensors every reader entry takes, checked once here rather than per read: tables_from
  // dereferences these tensors through raw pointers with no dtype or device check of its own. Run before
  // tables_from so a wrong-dtype or too-narrow table raises here, naming the tensor, not there.
  static void check_table_tensors(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions) {
    using namespace host;
    auto L_ = SymbolicSize{"layers"};
    auto E_ = SymbolicSize{"experts"};
    auto P_ = SymbolicSize{"parts"};
    auto cpu = SymbolicDevice{};
    verify_named("extents", TensorMatcher({L_, E_, P_, 4}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), extents);
    verify_named("starts", TensorMatcher({L_, E_}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), starts);
    verify_named("file_sizes", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), file_sizes);
    verify_named("segments", TensorMatcher({-1, 4}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), segments);
    verify_named("slabs", TensorMatcher({L_, kNumNames<Layout>}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slabs);
    verify_named(
        "row_bytes", TensorMatcher({kNumNames<Layout>}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), row_bytes);
    verify_named("buffer_regions", TensorMatcher({-1, 3}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), buffer_regions);
  }

  static std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
  }

  // Shared ownership: every call holds its own reference, so a close() from another Python
  // thread (or a finalizer) frees the service only after the calls in flight return.
  static std::unordered_map<int64_t, std::shared_ptr<Tier>>& registry() {
    static std::unordered_map<int64_t, std::shared_ptr<Tier>> tiers;
    return tiers;
  }

  static std::shared_ptr<Tier> find(int64_t handle) {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto found = registry().find(handle);
    if (found == registry().end()) throw std::runtime_error(error_prefix<Layout>() + "unknown handle");
    return found->second;
  }

  // Guarded by registry_mutex(), like the tiers; shared for the same reason as the tiers.
  static std::unordered_map<int64_t, std::shared_ptr<Thread>>& thread_registry() {
    static std::unordered_map<int64_t, std::shared_ptr<Thread>> threads;
    return threads;
  }

  static std::shared_ptr<Thread> find_thread(int64_t handle) {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto found = thread_registry().find(handle);
    if (found == thread_registry().end()) throw std::runtime_error(error_prefix<Layout>() + "no service thread");
    return found->second;
  }

  /// \brief The build policy this module was compiled with: "prod" or "instr" (build_policy.h).
  static std::string build_name() {
    return std::string(Build::kName);
  }

  /// \brief The layout this module was built for: its tensor names in copy-table order, newline-joined.
  static std::string layout_names() {
    std::string out;
    for (const auto name : Layout::kNames)
      out += std::string(name) + "\n";
    out.pop_back();
    return out;
  }

  /// \brief Bit i: name i may be read by the copy wait's SMs (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES).
  static int64_t layout_small_mask() {
    return Layout::kSmallMask;
  }

  // Read `experts` of streamed row `row` into `slots` once, synchronously (tests, tools).
  // Arguments are validated by the Python wrapper (read_rows_once).
  static int64_t read_rows(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
      int64_t direct,
      int64_t row,
      TensorView experts,
      TensorView slots,
      int64_t step) {
    using namespace host;
    check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
    auto cpu = SymbolicDevice{};
    verify_named("experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), experts);
    verify_named("slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slots);
    Source reader(
        tables_from<Layout>(
            extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images),
        direct != 0);
    if (!reader.open()) return 0;
    return reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), [](size_t) { return false; });
  }

  // The fault words `f` for a read available in both builds (read_rows_traced, read_rows_pieces): installed on
  // InstrBuild; on ProdBuild, which has no fault state, a tensor that injects one is refused (`what` names it) and an
  // inert one (only the non-fault words: abandon_after, step, piece_stream, the chunk and cut caps) is accepted.
  static void install_fault(Source& reader, const int64_t* f, const char* what) {
    if constexpr (Build::kFaults) {
      (void)what;
      reader.set_fault(fault_from(f));
    } else {
      (void)reader;
      if (injects_fault(f)) test_only(what);
    }
  }

  // Test only: expert_stream_read_rows with the reader's StageRecord copied to `record`
  // (stage_words() int64), with `ok` and `status` set from the result. `fault` is the faulted call's
  // tensor, laid out as expert_stream_read_rows_faulted's (kFaultWords words); a _fault_tensor() with no fault kwargs
  // injects nothing. (An all-zero tensor is not that: 0 in word 15 selects row 0 and in word 16 arms a hold; the
  // Python wrapper sends -1 in both.)
  // `owner_core` (test-only owner-pinning scaffold, PACK_WORKERS.md): -1 (the Python wrapper's default)
  // leaves the reader byte-for-byte what it is without this parameter; >= 0 pins the calling/owner thread
  // to that core and excludes it from the packing pool's mask (ReaderCore::set_owner_core).
  static int64_t read_rows_traced(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
      int64_t direct,
      int64_t row,
      TensorView experts,
      TensorView slots,
      int64_t step,
      TensorView fault,
      TensorView record,
      int64_t owner_core) {
    using namespace host;
    check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
    auto cpu = SymbolicDevice{};
    verify_named("fault", TensorMatcher({kFaultWords}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), fault);
    verify_named("experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), experts);
    verify_named("slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slots);
    verify_named("record", TensorMatcher({stage_words()}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), record);
    check_fault_words<Layout>(fault);
    const auto* f = static_cast<const int64_t*>(fault.data_ptr());
    Source reader(
        tables_from<Layout>(
            extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images),
        direct != 0);
    reader.set_owner_core(owner_core);
    if (f[22] != 0) reader.set_piece_stream(true);
    reader.set_fixed_chunk_cap(f[28]);
    reader.set_leg_cut_cap(f[31]);
    if (!reader.open()) return 0;
    install_fault(reader, f, "a fault on read_rows_traced");
    StageRecord stage;
    const int result =
        reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), abandon_after(f[17]), &stage);
    stage.ok = result == 1 ? 1 : 0;
    stage.status = result == 1 ? kStatusServed : result == 0 ? kStatusFailed : kStatusCancelled;
    std::memcpy(record.data_ptr(), &stage, sizeof(stage));
    return result;
  }

  // Test only: one reader reads `experts` into `slots` with `fault` injected
  // (see ReadFault and fault_from), then reads `then_experts` into `then_slots` with no fault (no second read when
  // there are none). Results go to
  // `results[0..11]`: the two reads' results, the completions the reader had reaped after each, then its
  // stale completions, generation wraps, the packing jobs still open when the first read returned, the
  // number of packing workers the reader has, and its fixed_cuts, fanout_sqes, cut_reads and gap_cuts after the first
  // read.
  static void read_rows_faulted(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
      int64_t direct,
      int64_t row,
      TensorView experts,
      TensorView slots,
      TensorView then_experts,
      TensorView then_slots,
      TensorView fault,
      TensorView results) {
    if constexpr (!Build::kFaults) {
      test_only("read_rows_faulted");
    } else {
      using namespace host;
      check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
      auto cpu = SymbolicDevice{};
      verify_named("fault", TensorMatcher({kFaultWords}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), fault);
      verify_named("experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), experts);
      verify_named("slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slots);
      verify_named("then_experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), then_experts);
      verify_named("then_slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), then_slots);
      verify_named("results", TensorMatcher({12}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), results);
      auto* out = static_cast<int64_t*>(results.data_ptr());
      check_fault_words<Layout>(fault);
      const auto* f = static_cast<const int64_t*>(fault.data_ptr());
      Source reader(
          tables_from<Layout>(
              extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images),
          direct != 0);
      if (f[22] != 0) reader.set_piece_stream(true);
      reader.set_fixed_chunk_cap(f[28]);
      reader.set_leg_cut_cap(f[31]);
      if (!reader.open()) {
        std::fill(out, out + 12, 0);
        return;
      }
      reader.set_fault(fault_from(f));
      const size_t step = f[18] > 0 ? static_cast<size_t>(f[18]) : static_cast<size_t>(kBounceRows);
      out[0] = reader.read(row, ids_of(experts), slots_of(slots), step, abandon_after(f[17]));
      out[2] = reader.cqes();
      out[4] = reader.stale_cqes();
      out[5] = reader.generation_wraps();
      out[6] = 0;  // reserved: formerly the packing jobs still open (the packed path is gone)
      out[7] = reader.pack_workers();
      out[8] = reader.fixed_cuts();
      out[9] = reader.fanout_sqes();
      out[10] = reader.cut_reads();
      out[11] = reader.gap_cuts();
      reader.set_fault(ReadFault{});
      if (then_experts.size(0) == 0) return;  // a test that only wants the first read's state
      out[1] = reader.read(row, ids_of(then_experts), slots_of(then_slots), kBounceRows, abandon_after(0));
      out[3] = reader.cqes();
    }
  }

  // Test only (U10): expert_stream_read_rows_traced's read, recording every SQE the reader prepared. `sqes` receives
  // up to sqes.size(0) rows of 4 int64 (file, offset, length, bounce byte offset), in preparation order; `info` 11
  // int64: the result, the SQE count, the descriptor count, the ring credit, the completions reaped, fixed_cuts,
  // fanout_sqes, cut_reads, gap_cuts, min_cut_bytes and leg_stride. `fault` as the faulted call's (word 22 turns piece streaming on, word 28 caps registered chunks).
  static void read_rows_sqes(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
      int64_t direct,
      int64_t row,
      TensorView experts,
      TensorView slots,
      int64_t step,
      TensorView fault,
      TensorView record,
      TensorView sqes,
      TensorView info) {
    if constexpr (!Build::kFaults) {
      test_only("read_rows_sqes");
    } else {
      using namespace host;
      check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
      auto cpu = SymbolicDevice{};
      verify_named("fault", TensorMatcher({kFaultWords}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), fault);
      verify_named("experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), experts);
      verify_named("slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slots);
      verify_named("record", TensorMatcher({stage_words()}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), record);
      verify_named("sqes", TensorMatcher({-1, 4}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), sqes);
      verify_named("info", TensorMatcher({11}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), info);
      check_fault_words<Layout>(fault);
      const auto* f = static_cast<const int64_t*>(fault.data_ptr());
      auto* out = static_cast<int64_t*>(info.data_ptr());
      std::fill(out, out + 7, 0);
      Source reader(
          tables_from<Layout>(
              extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images),
          direct != 0);
      if (f[22] != 0) reader.set_piece_stream(true);
      reader.set_fixed_chunk_cap(f[28]);
      reader.set_leg_cut_cap(f[31]);
      if (!reader.open()) return;
      reader.set_fault(fault_from(f));
      std::vector<typename Source::SqeRecord> log;
      reader.set_sqe_log(&log);
      StageRecord stage;
      const int result =
          reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), abandon_after(f[17]), &stage);
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
      out[5] = reader.fixed_cuts();
      out[6] = reader.fanout_sqes();
      out[7] = reader.cut_reads();
      out[8] = reader.gap_cuts();
      out[9] = reader.min_cut_bytes();
      out[10] = reader.leg_stride();
    }
  }

  // Test only (U8): the owner's publish primitive on one readiness word (`word`, one int64): 1 when it set `bit`.
  static int64_t publish_piece(TensorView word, int64_t generation, int64_t bit) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("word", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), word);
    return expert_stream::publish_piece(
               static_cast<uint64_t*>(word.data_ptr()), static_cast<uint64_t>(generation), static_cast<uint8_t>(bit))
               ? 1
               : 0;
  }

  // Test only (U2, U3, U6): expert_stream_read_rows_traced's read, publishing each row's pieces into its readiness
  // words: row ordinal o's are masks[o][0 .. masks.size(1)), under `generation` (the caller initialises them). When
  // `reference` is not empty (a slab pointer table shaped like `slabs`, holding row o at ref_slots[o]), a checker
  // thread polls the first word of every row while the read runs and, for each bit it sees set, compares the piece's
  // bytes in the destination slab with the reference: what a device that acquired the bit would copy. `info` 5 int64:
  // the result, the reader's refused publishes, the pieces checked, the pieces whose bytes differed, and the bits the
  // checker saw set before the read returned.
  static void read_rows_pieces(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
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
    using namespace host;
    check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
    auto cpu = SymbolicDevice{};
    verify_named("fault", TensorMatcher({kFaultWords}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), fault);
    verify_named("experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), experts);
    verify_named("slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slots);
    verify_named("record", TensorMatcher({stage_words()}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), record);
    verify_named("masks", TensorMatcher({-1, -1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), masks);
    verify_named("info", TensorMatcher({5}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), info);
    check_fault_words<Layout>(fault);
    const auto* f = static_cast<const int64_t*>(fault.data_ptr());
    auto* out = static_cast<int64_t*>(info.data_ptr());
    std::fill(out, out + 5, 0);
    const Tables t = tables_from<Layout>(
        extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images);
    const std::vector<int32_t> ids = ids_of(experts);
    const std::vector<int64_t> dest = slots_of(slots);
    const size_t lanes = static_cast<size_t>(masks.size(1));
    if (static_cast<size_t>(masks.size(0)) != ids.size() || lanes == 0 || lanes > static_cast<size_t>(kPieceTargets)) {
      throw std::runtime_error(error_prefix<Layout>() + "masks must be [rows, 1..8] readiness words");
    }
    auto* words = static_cast<uint64_t*>(masks.data_ptr());
    std::vector<PieceTarget> targets(ids.size());
    for (size_t o = 0; o < ids.size(); ++o) {
      for (size_t l = 0; l < lanes; ++l)
        targets[o].words[targets[o].count++] = words + o * lanes + l;
    }
    const PiecePublish publish{static_cast<uint64_t>(generation), targets.data()};
    Source reader(Tables(t), direct != 0);
    if (f[22] != 0) reader.set_piece_stream(true);
    reader.set_fixed_chunk_cap(f[28]);
    reader.set_leg_cut_cap(f[31]);
    if (!reader.open()) return;
    install_fault(reader, f, "a fault on read_rows_pieces");

    // The checker: the pieces' runs per row, then poll until the read returns, and once more after.
    const bool checking = reference.numel() > 0;
    if (checking) {
      verify_named("reference", TensorMatcher({-1, -1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), reference);
      verify_named("ref_slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), ref_slots);
    }
    const size_t count = t.segments.size();
    std::vector<PieceRun> runs(ids.size() * kPieces * count);
    const auto* ref_table = checking ? static_cast<const int64_t*>(reference.data_ptr()) : nullptr;
    const auto* ref_slot = checking ? static_cast<const int64_t*>(ref_slots.data_ptr()) : nullptr;
    if (checking) {
      for (size_t o = 0; o < ids.size(); ++o) {
        RowGeometry g;
        if (!row_geometry(t, static_cast<size_t>(row * t.experts + ids[o]), g, &runs[o * kPieces * count])) {
          throw std::runtime_error(error_prefix<Layout>() + "the checker cannot cut a row");
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
            const auto* ref_base = reinterpret_cast<const uint8_t*>(
                static_cast<intptr_t>(ref_table[row * static_cast<int64_t>(t.slabs[row].size()) + s.name]));
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
        while (reading.load(std::memory_order_acquire))
          check_pass(true);
      });
    }
    StageRecord stage;
    int result = 0;
    try {
      result = reader.read(
          row,
          ids,
          dest,
          static_cast<size_t>(step),
          abandon_after(f[17]),
          &stage,
          nullptr,
          SIZE_MAX,
          NoProgress{},
          &publish);
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

  // Test only (U1): the sub-reads and pieces the reader computes when it admits expert `expert` of streamed row `row`
  // (row_geometry). `subs`: kPieces rows of 6 int64 (file, offset, length, dest, part, k), in file order; `pieces`:
  // kPieces rows of 1 + 2 * segments int64: the dependency mask, then (dst_lo, dst_hi) per segment in segment
  // destination coordinates (dst + the run's bounds). Returns the sub-read count, or -1 when the row cannot be cut.
  static int64_t piece_geometry(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
      int64_t row,
      int64_t expert,
      TensorView subs,
      TensorView pieces) {
    using namespace host;
    check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
    auto cpu = SymbolicDevice{};
    verify_named("subs", TensorMatcher({kPieces, 6}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), subs);
    const Tables t = tables_from<Layout>(
        extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images);
    const size_t count = t.segments.size();
    verify_named(
        "pieces",
        TensorMatcher({kPieces, 1 + 2 * static_cast<int64_t>(count)}).with_dtype<int64_t>().with_device<kDLCPU>(cpu),
        pieces);
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

  // The stream kernel's piece table (piece-streaming plan 4.2, open question 6: one entry per (row, expert), computed
  // by the reader's own row_geometry so the device cannot disagree with it). `runs`: int32 [layers, experts, kPieces,
  // segments, 2], each run as (dst_lo, dst_hi) byte offsets into the segment's name row. A row the reader refuses to
  // cut gets empty runs; its read fails, so no device copy ever uses them. Returns how many rows were refused.
  static int64_t piece_runs(
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
      TensorView runs) {
    using namespace host;
    check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
    auto cpu = SymbolicDevice{};
    verify_named("runs", TensorMatcher({-1, -1, -1, -1, -1}).with_dtype<int32_t>().with_device<kDLCPU>(cpu), runs);
    const Tables t = tables_from<Layout>(
        extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images);
    const size_t count = t.segments.size();
    const size_t rows = static_cast<size_t>(t.layers * t.experts);
    if (runs.size(0) != t.layers || runs.size(1) != t.experts || runs.size(2) != kPieces ||
        runs.size(3) != static_cast<int64_t>(count) || runs.size(4) != 2) {
      throw std::runtime_error(error_prefix<Layout>() + "the piece-run table has the wrong shape");
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
          throw std::runtime_error(
              error_prefix<Layout>() + "a piece run ends past the int32 range of the stream kernel's table");
        }
        line[2 * k] = static_cast<int32_t>(segment.dst + piece[k].lo);
        line[2 * k + 1] = static_cast<int32_t>(segment.dst + piece[k].hi);
      }
    }
    return refused;
  }

  static int64_t open(
      TensorView page,
      TensorView slot_map,
      TensorView extents,
      TensorView starts,
      TensorView file_sizes,
      TensorView segments,
      TensorView slabs,
      TensorView row_bytes,
      TensorView buffer_regions,
      TensorView capacity,
      std::string paths,
      std::string source_paths,
      int64_t slot_bytes,
      int64_t row_images,
      int64_t direct,
      TensorView lease,
      TensorView hot_page) {
    using namespace host;
    check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
    // page, slot_map and lease are pinned (or not) together (ExpertStreamHost.__init__), so one SymbolicDevice
    // ties them to the same actual device; capacity is always a plain CPU tensor. hot_page is optional (an
    // absent one is an unpinned torch.empty(0)) and gets its own SymbolicDevice so it is not forced to equal
    // page's device when it is not given.
    auto host_mem = SymbolicDevice{};
    verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(host_mem), page);
    verify_named(
        "slot_map",
        TensorMatcher({extents.size(0), extents.size(1)})
            .with_dtype<int32_t>()
            .with_device<kDLCPU, kDLCUDAHost>(host_mem),
        slot_map);
    verify_named("lease", TensorMatcher({-1}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(host_mem), lease);
    auto hot_page_mem = SymbolicDevice{};
    verify_named(
        "hot_page", TensorMatcher({-1}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(hot_page_mem), hot_page);
    auto cpu = SymbolicDevice{};
    verify_named("capacity", TensorMatcher({extents.size(0)}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), capacity);
    const auto* capacity_data = static_cast<const int64_t*>(capacity.data_ptr());
    auto tier = std::make_shared<RamTier<Source>>(
        static_cast<uint8_t*>(page.data_ptr()),
        static_cast<int32_t*>(slot_map.data_ptr()),
        static_cast<uint8_t*>(lease.data_ptr()),
        lease.size(0),
        tables_from<Layout>(
            extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions, paths, source_paths, slot_bytes, row_images),
        std::vector<int64_t>(capacity_data, capacity_data + capacity.size(0)),
        direct != 0,
        hot_page.size(0) ? static_cast<uint8_t*>(hot_page.data_ptr()) : nullptr,
        hot_page.size(0));
    if (!tier->open()) return -1;
    std::lock_guard<std::mutex> guard(registry_mutex());
    static int64_t next_handle = 1;
    const int64_t handle = next_handle++;
    registry().emplace(handle, std::move(tier));
    return handle;
  }

  // 1 served a demand record, 3 a native-prefetch request, 2 an advisory record, 0 nothing posted. Refused while a
  // thread pumps. The order is the service thread's: demand, prefetch, advisory.
  static int64_t pump(int64_t handle) {
    const auto tier = find(handle);
    if (tier->threaded()) throw std::runtime_error(error_prefix<Layout>() + "pump() while the service thread runs");
    if (tier->pump_demand()) return 1;
    if (tier->pump_prefetch()) return 3;
    return tier->pump_advice() ? 2 : 0;
  }

  static int64_t contains(int64_t handle, int64_t row, int64_t expert) {
    return find(handle)->has(row, expert) ? 1 : 0;
  }

  static void touch(int64_t handle, int64_t row, int64_t expert) {
    find(handle)->touch(row, expert);
  }

  static void
  assign(int64_t handle, int64_t row, int64_t expert, TensorView protect, int64_t fallback, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("protect", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), protect);
    expert_stream::verify_named("out", TensorMatcher({2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    auto* result = static_cast<int64_t*>(out.data_ptr());
    int64_t evicted = -1;
    result[0] = find(handle)->assign(row, expert, expert_stream::ids_of(protect), fallback != 0, &evicted);
    result[1] = evicted;
  }

  // Prefill fills: `out` holds experts.size() + 1 int64, the claimed slots in order and then the evictions.
  static int64_t
  fill_begin(int64_t handle, int64_t row, TensorView experts, TensorView protect, int64_t fallback, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), experts);
    expert_stream::verify_named("protect", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), protect);
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    // Exact: fill_begin writes the evictions word at out[experts.size()] first, then a slot per claimed expert.
    expert_stream::verify_named(
        "out", TensorMatcher({experts.size(0) + 1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    auto* result = static_cast<int64_t*>(out.data_ptr());
    const std::vector<int32_t> ids = expert_stream::ids_of(experts);
    return find(handle)->fill_begin(
        row, ids, expert_stream::ids_of(protect), fallback != 0, result, result + ids.size());
  }

  static int64_t fill_wait(int64_t handle, int64_t rows, int64_t timeout_ns) {
    return find(handle)->fill_wait(rows, timeout_ns);
  }

  static int64_t fill_landed(int64_t handle) {
    return find(handle)->fill_landed();
  }

  static int64_t fill_end(int64_t handle) {
    return find(handle)->fill_end();
  }

  static void release(int64_t handle, int64_t row, int64_t slot) {
    find(handle)->release(row, slot);
  }

  static void slot_info(int64_t handle, int64_t row, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    // Exact: RamTier::slot_info writes 4 words per slot through a raw pointer with no bound.
    expert_stream::verify_named(
        "out", TensorMatcher({4 * tier->row_capacity(row)}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    tier->slot_info(row, static_cast<int64_t*>(out.data_ptr()));
  }

  static void lease_entry(int64_t handle, int64_t idx, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named(
        "out", TensorMatcher({4 + 3 * expert_stream::kLeaseLanes}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    if (idx < 0 || idx >= expert_stream::kDemandRecords)
      throw std::runtime_error(error_prefix<Layout>() + "request slot out of range");
    find(handle)->lease_entry(idx, static_cast<int64_t*>(out.data_ptr()));
  }

  static void inject_lease(int64_t handle, int64_t row, int64_t slot, int64_t delta) {
    if constexpr (!Build::kFaults) {
      test_only("inject_lease");
    } else {
      find(handle)->inject_lease(row, slot, delta);
    }
  }

  // out: free, evictable, leased.
  static void victim_census(int64_t handle, int64_t row, TensorView wanted, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("wanted", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), wanted);
    expert_stream::verify_named("out", TensorMatcher({3}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto census = find(handle)->victim_census(row, expert_stream::ids_of(wanted));
    auto* result = static_cast<int64_t*>(out.data_ptr());
    result[0] = census.free;
    result[1] = census.evictable;
    result[2] = census.leased;
  }

  // The watchdog's busy episode (D6): nonzero while a request or fill is in service, a new value per episode.
  static int64_t busy_episode(int64_t handle) {
    return static_cast<int64_t>(find(handle)->busy_episode());
  }

  static void close_admission(int64_t handle) {
    find(handle)->close_admission();
  }

  static void set_lease_mode(int64_t handle, int64_t on) {
    find(handle)->set_lease_mode(on != 0);
  }

  static void set_prefill_share(int64_t handle, int64_t share) {
    find(handle)->set_prefill_share(share);
  }

  static void set_gpu_hot(int64_t handle, int64_t on) {
    find(handle)->set_gpu_hot(on != 0);
  }

  static void set_two_phase(int64_t handle, int64_t on) {
    find(handle)->set_two_phase(on != 0);
  }

  static void set_piece_stream(int64_t handle, int64_t on) {
    find(handle)->set_piece_stream(on != 0);
  }

  static void enable_copy_engine(int64_t handle, int64_t device, int64_t spin_ns) {
    find(handle)->enable_copy_engine(device, spin_ns);
  }

  // entries: int64 [n, 3] of {source address, destination address, row bytes}; dst_rows: rows of every destination;
  // sm_mask: the entries the copy wait reads itself (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES), 0 for none.
  static void set_copy_table(int64_t handle, int64_t row, TensorView entries, int64_t dst_rows, int64_t sm_mask) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named(
        "entries", TensorMatcher({-1, 3}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), entries);
    if (entries.dim() != 2 || entries.size(1) != 3)
      throw std::runtime_error(error_prefix<Layout>() + "copy table must be [n, 3]");
    find(handle)->set_copy_table(
        row, static_cast<const int64_t*>(entries.data_ptr()), entries.size(0), dst_rows, sm_mask);
  }

  static void arm_copy_engine(int64_t handle, int64_t on) {
    find(handle)->arm_copy_engine(on != 0);
  }

  // page: pinned uint8 [kPrefetchPageBytes], the native-prefetch request and done lines.
  static void enable_native_prefetch(int64_t handle, TensorView page) {
    using namespace host;
    auto host_mem = SymbolicDevice{};
    expert_stream::verify_named(
        "page",
        TensorMatcher({expert_stream::kPrefetchPageBytes})
            .with_dtype<uint8_t>()
            .with_device<kDLCPU, kDLCUDAHost>(host_mem),
        page);
    if (page.dim() != 1 || page.size(0) != expert_stream::kPrefetchPageBytes)
      throw std::runtime_error(error_prefix<Layout>() + "the native prefetch page must be uint8 [256]");
    find(handle)->enable_native_prefetch(static_cast<uint8_t*>(page.data_ptr()));
  }

  // Test only: out int64 [3] = {active, row, slot} of the service's prefetch lease.
  static void prefetch_lease(int64_t handle, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({3}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    find(handle)->prefetch_lease(static_cast<int64_t*>(out.data_ptr()));
  }

  static int64_t copy_engine_idle(int64_t handle, int64_t timeout_ns) {
    return find(handle)->wait_copy_idle(expert_stream::now_ns() + timeout_ns) ? 1 : 0;
  }

  // Test only (HostCopyBackend): let `marks` more copy marks complete (negative: all), fail the calls, count marks.
  static void copy_engine_release(int64_t handle, int64_t marks) {
    find(handle)->host_copy_backend().release(marks);
  }

  static void copy_engine_fail(int64_t handle, int64_t issue, int64_t query) {
    if constexpr (!Build::kFaults) {
      test_only("copy_engine_fail");
    } else {
      find(handle)->host_copy_backend().fail(issue != 0, query != 0);
    }
  }

  // Test only: delay every copy job's completion by one extra copy of `bytes` from `src` to `dst` (0 bytes: off).
  static void copy_engine_ballast(int64_t handle, int64_t dst, int64_t src, int64_t bytes) {
    if constexpr (!Build::kFaults) {
      test_only("copy_engine_ballast");
    } else {
      find(handle)->copy_engine_ballast(static_cast<uint64_t>(dst), static_cast<uint64_t>(src), bytes);
    }
  }

  static int64_t copy_engine_marked(int64_t handle) {
    return find(handle)->host_copy_backend().marked();
  }

  static void inject_done_stall(int64_t handle, int64_t ns) {
    if constexpr (!Build::kFaults) {
      test_only("inject_done_stall");
    } else {
      find(handle)->inject_done_stall(ns);
    }
  }

  static void mapping(int64_t handle, int64_t row, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    tier->row_capacity(row);  // for its range check alone: RamTier::mapping indexes tiers_[row] unchecked
    // Exact: RamTier::mapping writes one word per expert through a raw pointer with no bound.
    expert_stream::verify_named(
        "out", TensorMatcher({tier->experts()}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    tier->mapping(row, static_cast<int64_t*>(out.data_ptr()));
  }

  static void slot_to_expert(int64_t handle, int64_t row, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    // Exact: RamTier::slot_to_expert writes one word per slot through a raw pointer with no bound.
    expert_stream::verify_named(
        "out", TensorMatcher({tier->row_capacity(row)}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    tier->slot_to_expert(row, static_cast<int64_t*>(out.data_ptr()));
  }

  static int64_t lru_order(int64_t handle, int64_t row, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    // Exact: RamTier::lru_order writes up to one word per slot (its READY slots) with no bound.
    expert_stream::verify_named(
        "out", TensorMatcher({tier->row_capacity(row)}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    return tier->lru_order(row, static_cast<int64_t*>(out.data_ptr()));
  }

  static void set_hot(int64_t handle, int64_t row, TensorView experts) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("experts", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), experts);
    find(handle)->set_hot(row, static_cast<const int64_t*>(experts.data_ptr()), experts.size(0));
  }

  // Test only (RamTier::inject): InstrBuild only.
  static void
  inject(int64_t handle, int64_t delay_ns, int64_t fail_reads, int64_t after_demands, int64_t abandon_after_batches) {
    if constexpr (!Build::kFaults) {
      test_only("inject");
    } else {
      find(handle)->inject(delay_ns, fail_reads != 0, after_demands, abandon_after_batches);
    }
  }

  // Test only: a full ReadFault for the tier's reader (the reader tests' fault tensor; see RamTier::inject_fault).
  static void inject_fault(int64_t handle, TensorView fault) {
    if constexpr (!Build::kFaults) {
      test_only("inject_fault");
    } else {
      using namespace host;
      auto cpu = SymbolicDevice{};
      expert_stream::verify_named(
          "fault", TensorMatcher({expert_stream::kFaultWords}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), fault);
      expert_stream::check_fault_words<Layout>(fault);
      find(handle)->inject_fault(static_cast<const int64_t*>(fault.data_ptr()));
    }
  }

  static void counters(int64_t handle, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named(
        "out", TensorMatcher({expert_stream::kCounterCount}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    find(handle)->counters(static_cast<int64_t*>(out.data_ptr()));
  }

  // Bit k set: counter k is a core counter (is_core_counter), kept by the production build. Python's CORE_COUNTERS
  // is checked against it.
  static int64_t core_counter_mask() {
    static_assert(expert_stream::kCounterCount <= 63, "the core-counter mask is one int64");
    int64_t mask = 0;
    for (int k = 0; k < expert_stream::kCounterCount; ++k)
      mask |= expert_stream::is_core_counter(k) ? int64_t{1} << k : 0;
    return mask;
  }

  static void layer_rows(int64_t handle, int64_t advisory, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    // Exact: RamTier::layer_rows writes one word per streamed layer through a raw pointer with no bound.
    expert_stream::verify_named(
        "out", TensorMatcher({tier->layers()}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    tier->layer_rows(static_cast<int64_t*>(out.data_ptr()), advisory != 0);
  }

  static int64_t trace_words() {
    return expert_stream::stage_words();
  }

  static void trace_enable(int64_t handle, int64_t capacity) {
    if (capacity <= 0) throw std::runtime_error(error_prefix<Layout>() + "the stage trace needs a positive capacity");
    find(handle)->enable_trace(static_cast<size_t>(capacity));
  }

  // Fills up to out.size(0) records, stage_words() int64 each; returns the count.
  static int64_t trace_drain(int64_t handle, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    // drain_trace treats `out` as a contiguous StageRecord array of out.size(0) records (it never reads
    // out.size(1)), so the row width must equal stage_words() int64 words or the stride is wrong.
    expert_stream::verify_named(
        "out", TensorMatcher({-1, expert_stream::stage_words()}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    return find(handle)->drain_trace(static_cast<expert_stream::StageRecord*>(out.data_ptr()), out.size(0));
  }

  // Test only: InstrBuild only (ProdBuild has no trace, so nothing to count). Gated on kMetrics, the trace's own flag.
  static int64_t trace_clock_reads() {
    if constexpr (!Build::kMetrics) {
      test_only("trace_clock_reads");
    } else {
      return expert_stream::traced_clock_reads().load(std::memory_order_relaxed);
    }
  }

  static int64_t trace_dropped(int64_t handle) {
    return find(handle)->trace_dropped();
  }

  // ---- Host-side simulated device: the post and wait kernels' protocol, for CPU tests ----

  static int64_t sim_post(
      TensorView page,
      int64_t row,
      TensorView need,
      TensorView protect,
      int64_t advisory,
      int64_t after,
      int64_t armed,
      int64_t lanes) {
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
  static int64_t sim_wait(TensorView page, int64_t seq, int64_t timeout_ns) {
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
  static void seqlock_stress(int64_t duration_ns, TensorView out) {
    if constexpr (!Build::kFaults) {
      test_only("seqlock_stress");
    } else {
      {
        using namespace host;
        auto cpu = SymbolicDevice{};
        expert_stream::verify_named("out", TensorMatcher({2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
      }
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
  }

  static void start_thread(int64_t handle, int64_t cpu_core, int64_t fatal_wait_ns, int64_t spin_ns) {
    if (cpu_core >= CPU_SETSIZE) throw std::runtime_error(error_prefix<Layout>() + "cpu_core out of range");
    if (cpu_core >= 64 && cpu_core <= 71) {
      throw std::runtime_error(
          error_prefix<Layout>() + "cores 64-71 are reserved (NVMe completion interrupts are pinned there)");
    }
    if (cpu_core < 0) {
      cpu_set_t inherited;
      CPU_ZERO(&inherited);
      if (pthread_getaffinity_np(pthread_self(), sizeof(inherited), &inherited) == 0) {
        for (int core = 64; core <= 71; ++core) {
          if (CPU_ISSET(core, &inherited)) {
            std::fprintf(
                stderr,
                "WARNING %sthe service thread inherits an affinity that includes reserved cores 64-71; "
                "run under taskset -c 0-63 or pass cpu_core\n",
                error_prefix<Layout>().c_str());
            break;
          }
        }
      }
    }
    std::shared_ptr<RamTier<Source>> tier = find(handle);
    // Checked and registered under one lock, so a concurrent close() either sees the thread
    // (and joins it) or runs before it and leaves no handle to start it on.
    std::lock_guard<std::mutex> guard(registry_mutex());
    if (registry().count(handle) == 0) throw std::runtime_error(error_prefix<Layout>() + "unknown handle");
    if (thread_registry().count(handle))
      throw std::runtime_error(error_prefix<Layout>() + "the service thread already runs");
    auto thread = std::make_shared<Thread>(std::move(tier), static_cast<int>(cpu_core), fatal_wait_ns, spin_ns);
    thread->start();
    thread_registry()[handle] = std::move(thread);
  }

  static void stop_thread(int64_t handle) {
    std::shared_ptr<Thread> thread;
    std::shared_ptr<Tier> tier;
    {
      std::lock_guard<std::mutex> guard(registry_mutex());
      const auto found = thread_registry().find(handle);
      if (found == thread_registry().end()) return;
      thread = std::move(found->second);
      thread_registry().erase(found);
      const auto owner = registry().find(handle);
      if (owner != registry().end()) tier = owner->second;
    }
    thread->stop();
    // The final settle (LEASE_PROTOCOL.md 7.5): no demand follows the last one to settle it, so a late second signal
    // on it is compared here, before ExpertStreamHost.stop writes its counters line. Here and not in RamThread::stop,
    // which ~RamThread also runs and which could then read a lease page the Python side already freed; at this point
    // the thread has joined and the page is still alive, because close() runs after this call.
    if (tier) tier->retire_leases(true);
  }

  static int64_t pause(int64_t handle, int64_t timeout_ns) {
    return find_thread(handle)->pause(timeout_ns);
  }

  static void resume(int64_t handle) {
    find_thread(handle)->resume();
  }

  // Takes the tier and its service thread out of the registries under one lock (so no
  // start_thread can slip in between), then joins the thread: it holds a reference to the
  // tier, which writes through raw addresses of Python-owned tensors that the caller
  // releases after this returns.
  static void close(int64_t handle) {
    std::shared_ptr<Thread> thread;
    std::shared_ptr<RamTier<Source>> tier;
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
};

}  // namespace sglang::expert_stream

// One line per export; the list is the module's whole Python-visible surface.
#define EXPERT_STREAM_HOST_EXPORTS(Exports)                                                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_build_name, Exports::build_name);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_layout_names, Exports::layout_names);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_layout_small_mask, Exports::layout_small_mask);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows, Exports::read_rows);                           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_traced, Exports::read_rows_traced);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_faulted, Exports::read_rows_faulted);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_sqes, Exports::read_rows_sqes);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_publish_piece, Exports::publish_piece);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_pieces, Exports::read_rows_pieces);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_piece_geometry, Exports::piece_geometry);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_piece_runs, Exports::piece_runs);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_open, Exports::open);                                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_close, Exports::close);                                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pump, Exports::pump);                                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_contains, Exports::contains);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_touch, Exports::touch);                                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_assign, Exports::assign);                                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_release, Exports::release);                               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_begin, Exports::fill_begin);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_wait, Exports::fill_wait);                           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_landed, Exports::fill_landed);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_end, Exports::fill_end);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_slot_info, Exports::slot_info);                           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject_lease, Exports::inject_lease);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_lease_entry, Exports::lease_entry);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_victim_census, Exports::victim_census);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_busy_episode, Exports::busy_episode);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_close_admission, Exports::close_admission);               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_lease_mode, Exports::set_lease_mode);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_gpu_hot, Exports::set_gpu_hot);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_prefill_share, Exports::set_prefill_share);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_two_phase, Exports::set_two_phase);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_piece_stream, Exports::set_piece_stream);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_enable_copy_engine, Exports::enable_copy_engine);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_copy_table, Exports::set_copy_table);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_arm_copy_engine, Exports::arm_copy_engine);               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_idle, Exports::copy_engine_idle);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_enable_native_prefetch, Exports::enable_native_prefetch); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_prefetch_lease, Exports::prefetch_lease);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_release, Exports::copy_engine_release);       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_fail, Exports::copy_engine_fail);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_marked, Exports::copy_engine_marked);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_ballast, Exports::copy_engine_ballast);       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject_done_stall, Exports::inject_done_stall);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_mapping, Exports::mapping);                               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_slot_to_expert, Exports::slot_to_expert);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_lru_order, Exports::lru_order);                           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_hot, Exports::set_hot);                               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject, Exports::inject);                                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject_fault, Exports::inject_fault);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_counters, Exports::counters);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_core_counter_mask, Exports::core_counter_mask);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_layer_rows, Exports::layer_rows);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_words, Exports::trace_words);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_enable, Exports::trace_enable);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_drain, Exports::trace_drain);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_dropped, Exports::trace_dropped);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_clock_reads, Exports::trace_clock_reads);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_sim_post, Exports::sim_post);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_sim_wait, Exports::sim_wait);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_seqlock_stress, Exports::seqlock_stress);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_start_thread, Exports::start_thread);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_stop_thread, Exports::stop_thread);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pause, Exports::pause);                                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_resume, Exports::resume);
