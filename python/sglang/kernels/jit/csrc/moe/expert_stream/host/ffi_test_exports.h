// The expert-stream host exports the server does not call: tests use them, and analysis tools call some of them on a
// live module.
//
// Every site that expands EXPERT_STREAM_HOST_EXPORTS(Exports) also expands EXPERT_STREAM_HOST_TEST_EXPORTS(Exports),
// the production build included. Exports that need the fault or trace state refuse in the production build
// (test_only()); the rest (reading, geometry, counters) work in both, and tools use some of them on a live module.
//
//   reader     read_rows, read_rows_traced, read_rows_faulted, read_rows_sqes, read_rows_pieces, piece_geometry,
//              publish_piece: one synchronous read through the reader, with traces and injected faults
//   tier       pump, pump_group, slot_info, handled_through, victim_census, busy_episode, inject, inject_fault, trace_clock_reads
//   copy       copy_engine_idle, copy_engine_release, copy_engine_fail, copy_engine_marked, copy_engine_ballast
//   protocol   seqlock_stress, read_record_fields
//   misc       test_kernel_address, test_kernel_calls, test_kernel_hold, test_keep_warm_calls, test_keep_warm_core,
//              pause_ns
//   draft      draft_test_post, draft_test_tear, draft_test_finish_close: the draft channel's device half on the host;
//              draft_test_poll_pause: the draft CPU thread's poll path sleeps between its stop and head loads
//   kernel     kernel_layer, kernel_forward, kernel_error, kernel_drop: any kernel's make_layer and forward, by layer id;
//              in both builds, as the DSpark draft's CPU experts call them (cpu_experts/draft.py)
//
// Arguments are validated by the Python wrappers in
// python/sglang/kernels/ops/moe/expert_stream_transport.py.
#pragma once

#include "ffi_exports.h"
#include <algorithm>
#include <array>
#include <mutex>
#include <span>
#include <vector>

namespace sglang::expert_stream {

template <class Exports>
struct HostTestExports;

/// \brief The test and tool host exports of one HostExports instantiation.
///
/// Derived from HostExports, so a handle its open() returned resolves here: both use the same function-local
/// registries.
template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build>
struct HostTestExports<HostExports<Layout, Reader, Build>> : HostExports<Layout, Reader, Build> {
  using Base = HostExports<Layout, Reader, Build>;
  using Base::check_table_tensors;
  using Base::find;
  using typename Base::Source;

  // Copies a slots tensor into a vector.
  static std::vector<int64_t> slots_of(TensorView slots) {
    const auto* data = static_cast<const int64_t*>(slots.data_ptr());
    return std::vector<int64_t>(data, data + slots.size(0));
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
            extents,
            starts,
            file_sizes,
            segments,
            slabs,
            row_bytes,
            buffer_regions,
            paths,
            source_paths,
            slot_bytes,
            row_images),
        direct != 0);
    if (!reader.open()) return 0;
    return reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), [](size_t) { return false; });
  }

  // Applies the fault words `f` for a read that exists in both builds (read_rows_traced, read_rows_pieces). InstrBuild
  // installs them. ProdBuild has no fault state, so it refuses a tensor that injects a fault (`what` names the export)
  // and accepts an inert one (only the non-fault words: abandon_after, step, piece_stream, the chunk and cut caps).
  static void install_fault(Source& reader, const int64_t* f, const char* what) {
    if constexpr (Build::kFaults) {
      (void)what;
      reader.set_fault(fault_from(f));
    } else {
      (void)reader;
      if (injects_fault(f)) test_only(what);
    }
  }

  // Test only: expert_stream_read_rows with the reader's StageRecord copied to `record` (stage_words() int64), with
  // `ok` and `status` set from the result. `fault` is laid out as expert_stream_read_rows_faulted's (kFaultWords
  // words); a _fault_tensor() with no fault kwargs injects nothing. (An all-zero tensor is not that: 0 in word 15
  // selects row 0 and in word 16 arms a hold; the Python wrapper sends -1 in both.)
  //
  // `owner_core` is a test-only owner-pinning scaffold (analysis/dsv41-drive/PACK_WORKERS.md): -1 (the Python
  // wrapper's default) leaves the reader unchanged; >= 0 pins the calling/owner thread to that core and excludes it
  // from the packing pool's mask (ReaderCore::set_owner_core).
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
            extents,
            starts,
            file_sizes,
            segments,
            slabs,
            row_bytes,
            buffer_regions,
            paths,
            source_paths,
            slot_bytes,
            row_images),
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
              extents,
              starts,
              file_sizes,
              segments,
              slabs,
              row_bytes,
              buffer_regions,
              paths,
              source_paths,
              slot_bytes,
              row_images),
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

  // Test only: expert_stream_read_rows_traced's read, recording every SQE the reader prepared. `sqes` receives
  // up to sqes.size(0) rows of 4 int64 (file, offset, length, bounce byte offset), in preparation order; `info` 11
  // int64: the result, the SQE count, the descriptor count, the ring credit, the completions reaped, fixed_cuts,
  // fanout_sqes, cut_reads, gap_cuts, min_cut_bytes and leg_stride. `fault` as the faulted call's (word 22 turns piece
  // streaming on, word 28 caps registered chunks).
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
              extents,
              starts,
              file_sizes,
              segments,
              slabs,
              row_bytes,
              buffer_regions,
              paths,
              source_paths,
              slot_bytes,
              row_images),
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

  // Test only: the owner's publish primitive on one readiness word (`word`, one int64): 1 when it set `bit`.
  static int64_t publish_piece(TensorView word, int64_t generation, int64_t bit) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("word", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), word);
    return expert_stream::publish_piece(
               static_cast<uint64_t*>(word.data_ptr()), static_cast<uint64_t>(generation), static_cast<uint8_t>(bit))
               ? 1
               : 0;
  }

  // Test only: expert_stream_read_rows_traced's read, publishing each row's pieces into its readiness
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
        extents,
        starts,
        file_sizes,
        segments,
        slabs,
        row_bytes,
        buffer_regions,
        paths,
        source_paths,
        slot_bytes,
        row_images);
    const std::vector<int32_t> ids = ids_of(experts);
    const std::vector<int64_t> dest = slots_of(slots);
    const size_t lanes = static_cast<size_t>(masks.size(1));
    if (static_cast<size_t>(masks.size(0)) != ids.size() || lanes == 0 || lanes > static_cast<size_t>(kPieceTargets)) {
      throw std::runtime_error(
          error_prefix<Layout>() + "masks must be [rows, 1.." + std::to_string(kPieceTargets) + "] readiness words");
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

  // Test only: the sub-reads and pieces the reader computes when it admits expert `expert` of streamed row `row`
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
        extents,
        starts,
        file_sizes,
        segments,
        slabs,
        row_bytes,
        buffer_regions,
        paths,
        source_paths,
        slot_bytes,
        row_images);
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

  // Serves one demand record per group on the calling thread: 1 if group 0 served one, 0 if nothing was posted (or it
  // was deferred).
  // Throws while the service thread runs. Held under caller_mutex(): pump() consumes the copy-completion ring (and owns
  // the tier), so it is serialized against every other Python caller, whose owned calls and wait_copy_idle drain the
  // same ring. Tests only; no hot-path cost.
  static int64_t pump(int64_t handle) {
    const auto tier = find(handle);
    std::lock_guard<std::mutex> caller(tier->caller_mutex());
    if (tier->threaded()) throw std::runtime_error(error_prefix<Layout>() + "pump() while the service thread runs");
    return tier->pump_demand() ? 1 : 0;
  }

  // Serves one demand record of group `group` alone, on the calling thread: 1 if it served one. Under the same rules as
  // pump(); the other groups stay where they are, so a test can leave one group behind the device.
  static int64_t pump_group(int64_t handle, int64_t group) {
    if constexpr (!Build::kFaults) {
      (void)handle, (void)group;
      test_only("pump_group");
    } else {
      const auto tier = find(handle);
      std::lock_guard<std::mutex> caller(tier->caller_mutex());
      if (tier->threaded())
        throw std::runtime_error(error_prefix<Layout>() + "pump_group() while the service thread runs");
      if (group < 0 || group >= tier->groups())
        throw std::runtime_error(error_prefix<Layout>() + "group " + std::to_string(group) + " is out of range");
      return tier->pump_demand(static_cast<int>(group)) ? 1 : 0;
    }
  }

  // Fills `out` with RamTier::slot_info's three words per slot of `row`.
  static void slot_info(int64_t handle, int64_t row, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    // Exact: RamTier::slot_info writes 3 words per slot through a raw pointer with no bound.
    expert_stream::verify_named(
        "out", TensorMatcher({3 * tier->row_capacity(row)}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    tier->slot_info(row, static_cast<int64_t*>(out.data_ptr()));
  }

  // The last seq the service finished (what ChainSim.wait_handled waits on).
  static int64_t handled_through(int64_t handle) {
    return static_cast<int64_t>(find(handle)->handled_through());
  }

  // Counts the free and evictable slots of `row` for a request that routes `wanted`: out = {free, evictable}.
  static void victim_census(int64_t handle, int64_t row, TensorView wanted, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("wanted", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), wanted);
    expert_stream::verify_named("out", TensorMatcher({2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto census = find(handle)->victim_census(row, expert_stream::ids_of(wanted));
    auto* result = static_cast<int64_t*>(out.data_ptr());
    result[0] = census.free;
    result[1] = census.evictable;
  }

  // Group 0's busy episode (the watchdog's per group): nonzero while a request or fill is in service, a new value per
  // episode.
  static int64_t busy_episode(int64_t handle) {
    return static_cast<int64_t>(find(handle)->busy_episode(0));
  }

  // 1 when every job handed to the copy thread completed or failed within `timeout_ns`, else 0.
  static int64_t copy_engine_idle(int64_t handle, int64_t timeout_ns) {
    return find(handle)->wait_copy_idle(expert_stream::now_ns() + timeout_ns) ? 1 : 0;
  }

  // Test only: the CPU expert kernel tests enable in place of a format's (test_kernel_address). Its layers read only
  // their hidden (and capacity, which a call records). A forward spins k * ns_per_expert, waits while its worker-0 core
  // is held (test_kernel_hold), throws when made failing, else for each row t < max(rows, 1) writes out[t * hidden + j]
  // for j < max(hidden, 1) -- (accumulate ? out[..] : j) + sum_i weights[t * k + i] * (slots[t * k + i] + 1), or a zero
  // partial (accumulate ? out[..] : 0) when made zeroing -- and records one call per row. A keep-warm counts its calls
  // and records its first core, then spins until its word moves or its deadline passes.
  class FakeKernel final : public cpu_experts::CpuExpertKernel {
   public:
    struct Call {
      int32_t core, affinity, threads, accumulate, k;
      std::array<int32_t, Wire::kLanes> slots;
      std::array<float, Wire::kLanes> weights;
      int32_t capacity;  // the layer's: which layer the call ran
    };

    void reset(int64_t ns_per_expert, int64_t fail, bool zero) {
      std::lock_guard<std::mutex> lock(mutex_);
      calls_.clear();
      ns_.store(ns_per_expert, std::memory_order_relaxed);
      fail_.store(fail, std::memory_order_relaxed);
      zero_.store(zero, std::memory_order_relaxed);
      held_core_.store(-1, std::memory_order_release);
      warm_calls_.store(0, std::memory_order_relaxed);
      warm_core_.store(-1, std::memory_order_relaxed);
    }

    const char* name() const noexcept override {
      return "fake";
    }
    cpu_experts::ExpertLayer make_layer(
        const cpu_experts::ExpertLayer& shape, std::span<const std::byte>) const override {
      cpu_experts::ExpertLayer layer = shape;
      layer.kernel = this;
      return layer;
    }
    int32_t max_routes() const noexcept override {
      return Wire::kLanes;
    }
    int32_t max_rows() const noexcept override {
      return 1 << 16;
    }
    void check(const cpu_experts::ExpertLayer& layer, const cpu_experts::ForwardCall&) const override {
      if (layer.kernel != this) throw std::invalid_argument("fake CPU expert kernel: another kernel's layer");
    }
    void forward(const cpu_experts::ExpertLayer& layer, const cpu_experts::ForwardCall& c) const override {
      const int32_t core = c.cores.empty() ? -1 : c.cores.front();
      const int64_t until = now_ns() + c.k * ns_.load(std::memory_order_relaxed);
      while (now_ns() < until)
        _mm_pause();
      while (core >= 0 && held_core_.load(std::memory_order_acquire) == core)
        _mm_pause();
      if (const int64_t f = fail_.load(std::memory_order_relaxed); f != 0)
        throw std::runtime_error("fake CPU expert forward failed (" + std::to_string(f) + ")");
      const int32_t hidden = std::max(layer.hidden, 1);
      const bool zero = zero_.load(std::memory_order_relaxed);
      for (int32_t t = 0; t < std::max(c.rows, 1); ++t) {
        const int32_t* slots = c.slots + static_cast<int64_t>(t) * c.k;
        const float* weights = c.weights + static_cast<int64_t>(t) * c.k;
        float* out = c.out + static_cast<int64_t>(t) * hidden;
        double sum = 0;
        for (int32_t i = 0; i < c.k; ++i)
          sum += static_cast<double>(weights[i]) * (slots[i] + 1);
        for (int32_t j = 0; j < hidden; ++j) {
          const double base = c.accumulate ? static_cast<double>(out[j]) : (zero ? 0.0 : static_cast<double>(j));
          out[j] = static_cast<float>(zero ? base : base + sum);
        }
      }
      cpu_set_t mask;
      CPU_ZERO(&mask);
      int32_t affinity = -1;
      if (sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_COUNT(&mask) == 1)
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
          if (CPU_ISSET(cpu, &mask)) affinity = cpu;
      std::lock_guard<std::mutex> lock(mutex_);
      for (int32_t t = 0; t < std::max(c.rows, 1); ++t) {
        Call call{core, affinity, c.threads, c.accumulate ? 1 : 0, std::min<int32_t>(c.k, Wire::kLanes), {}, {},
                  layer.capacity};
        for (int32_t i = 0; i < call.k; ++i) {
          call.slots[i] = c.slots[static_cast<int64_t>(t) * c.k + i];
          call.weights[i] = c.weights[static_cast<int64_t>(t) * c.k + i];
        }
        calls_.push_back(call);
      }
    }
    void keep_warm(std::span<const int> cores, int32_t, const uint32_t* word, uint32_t seen, int64_t,
                   int64_t release_ns) const override {
      warm_calls_.fetch_add(1, std::memory_order_relaxed);
      warm_core_.store(cores.empty() ? -1 : cores.front(), std::memory_order_relaxed);
      while (__atomic_load_n(word, __ATOMIC_ACQUIRE) == seen && now_ns() < release_ns)
        _mm_pause();
    }

    std::vector<Call> calls() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return calls_;
    }
    void hold(int64_t core, bool on) {
      held_core_.store(on ? core : -1, std::memory_order_release);
    }
    int64_t warm_calls() const {
      return warm_calls_.load(std::memory_order_relaxed);
    }
    int64_t warm_core() const {
      return warm_core_.load(std::memory_order_relaxed);
    }

   private:
    mutable std::mutex mutex_;
    mutable std::vector<Call> calls_;
    std::atomic<int64_t> ns_{0}, fail_{0}, held_core_{-1};
    std::atomic<bool> zero_{false};
    mutable std::atomic<int64_t> warm_calls_{0}, warm_core_{-1};
  };
  static FakeKernel& fake_kernel() {
    static FakeKernel kernel;
    return kernel;
  }

  // Test only: the fake kernel's address, its calls and keep-warm counts reset; `fail` nonzero makes every forward
  // throw, `zero` makes it write a zero partial.
  static int64_t test_kernel_address(int64_t ns_per_expert, int64_t fail, int64_t zero) {
    if constexpr (!Build::kFaults) {
      test_only("test_kernel_address");
    } else {
      fake_kernel().reset(ns_per_expert, fail, zero != 0);
      return static_cast<int64_t>(reinterpret_cast<intptr_t>(&fake_kernel()));
    }
  }
  // Test only: the fake's calls since test_kernel_address, one per row, as float64 rows {core, affinity, threads,
  // accumulate, k, slots[kLanes], weights[kLanes], capacity} into `out` (as many as fit); returns how many there are.
  static int64_t test_kernel_calls(TensorView out) {
    if constexpr (!Build::kFaults) {
      test_only("test_kernel_calls");
    } else {
      using namespace host;
      auto cpu = SymbolicDevice{};
      expert_stream::verify_named(
          "out", TensorMatcher({-1, 6 + 2 * Wire::kLanes}).with_dtype<double>().with_device<kDLCPU>(cpu), out);
      const std::vector<typename FakeKernel::Call> calls = fake_kernel().calls();
      auto* o = static_cast<double*>(out.data_ptr());
      const int64_t width = 6 + 2 * Wire::kLanes;
      for (int64_t r = 0; r < std::min<int64_t>(out.size(0), static_cast<int64_t>(calls.size())); ++r) {
        const typename FakeKernel::Call& c = calls[r];
        double* row = o + r * width;
        row[0] = c.core;
        row[1] = c.affinity;
        row[2] = c.threads;
        row[3] = c.accumulate;
        row[4] = c.k;
        for (int i = 0; i < Wire::kLanes; ++i) {
          row[5 + i] = c.slots[i];
          row[5 + Wire::kLanes + i] = c.weights[i];
        }
        row[5 + 2 * Wire::kLanes] = c.capacity;
      }
      return static_cast<int64_t>(calls.size());
    }
  }
  // Test only: while on, a fake forward whose worker-0 core is `core` waits (one held core at a time).
  static void test_kernel_hold(int64_t core, int64_t on) {
    if constexpr (!Build::kFaults) {
      test_only("test_kernel_hold");
    } else {
      fake_kernel().hold(core, on != 0);
    }
  }
  static int64_t test_keep_warm_calls() {
    if constexpr (!Build::kFaults) {
      test_only("test_keep_warm_calls");
    } else {
      return fake_kernel().warm_calls();
    }
  }
  // Test only: the first core the fake keep-warm's last call took (-1 before any call since test_kernel_address).
  static int64_t test_keep_warm_core() {
    if constexpr (!Build::kFaults) {
      test_only("test_keep_warm_core");
    } else {
      return fake_kernel().warm_core();
    }
  }

  // Test only: the draft channel's device half on the host (draft_kernels.cuh), at the pinned channel's address.
  // draft_test_post writes what draft_post_kernel's thread 0 writes, in its order: the record's seqlock open, the
  // stage|rows|k word and the epoch, the seq, then the head.
  static void draft_test_post(int64_t address, int64_t stage, int64_t rows, int64_t k, int64_t seq, int64_t epoch) {
    if constexpr (!Build::kFaults) {
      test_only("draft_test_post");
    } else {
      using S = draft::DraftChannel;
      auto* ch = reinterpret_cast<uint8_t*>(static_cast<intptr_t>(address));
      const auto s = static_cast<uint32_t>(seq);
      auto* rec = reinterpret_cast<uint32_t*>(ch + S::kRing + channel::ring_index<S>(s) * S::kRecordBytes);
      __atomic_store_n(rec, 0u, __ATOMIC_RELAXED);
      std::atomic_thread_fence(std::memory_order_release);
      __atomic_store_n(
          rec + draft::kRecStage / 4,
          (static_cast<uint32_t>(stage) & 0xFFFFu) | (static_cast<uint32_t>(rows) & 0xFFu) << 16 |
              (static_cast<uint32_t>(k) & 0xFFu) << 24,
          __ATOMIC_RELAXED);
      __atomic_store_n(rec + draft::kRecEpoch / 4, static_cast<uint32_t>(epoch), __ATOMIC_RELAXED);
      __atomic_store_n(rec, s, __ATOMIC_RELEASE);
      __atomic_store_n(reinterpret_cast<uint32_t*>(ch + S::kHead), s, __ATOMIC_RELEASE);
    }
  }
  // Test only: publishes head `seq` over a record whose seq word reads 0 (a torn record).
  static void draft_test_tear(int64_t address, int64_t seq) {
    if constexpr (!Build::kFaults) {
      test_only("draft_test_tear");
    } else {
      using S = draft::DraftChannel;
      auto* ch = reinterpret_cast<uint8_t*>(static_cast<intptr_t>(address));
      const auto s = static_cast<uint32_t>(seq);
      __atomic_store_n(
          reinterpret_cast<uint32_t*>(ch + S::kRing + channel::ring_index<S>(s) * S::kRecordBytes), 0u, __ATOMIC_RELEASE);
      __atomic_store_n(reinterpret_cast<uint32_t*>(ch + S::kHead), s, __ATOMIC_RELEASE);
    }
  }
  // Test only: the device's close_gate on the host: close the gate for G, fence, re-check done[G] and open the gate
  // itself if it is there. Returns 1 when it opened the gate.
  static int64_t draft_test_finish_close(int64_t address, int64_t seq, int64_t epoch) {
    if constexpr (!Build::kFaults) {
      test_only("draft_test_finish_close");
    } else {
      using S = draft::DraftChannel;
      auto* ch = reinterpret_cast<uint8_t*>(static_cast<intptr_t>(address));
      const auto s = static_cast<uint32_t>(seq);
      const uint64_t gen = static_cast<uint64_t>(static_cast<uint32_t>(epoch)) << 32 | s;
      auto* gate = reinterpret_cast<uint32_t*>(ch + S::kGate);
      __atomic_store_n(gate, channel::gate_word(s, channel::kGateClosed), __ATOMIC_RELAXED);
      std::atomic_thread_fence(std::memory_order_seq_cst);
      const auto* done = reinterpret_cast<const uint64_t*>(ch + S::kDone + channel::ring_index<S>(s) * S::kDoneBytes);
      if (__atomic_load_n(done, __ATOMIC_ACQUIRE) != gen) return 0;
      __atomic_store_n(gate, channel::gate_word(s, channel::kGateOpen), __ATOMIC_RELEASE);
      return 1;
    }
  }

  // Test only: the draft CPU thread's poll path sleeps `us` microseconds (0: not at all) between loading its stop flag
  // and the head word, which holds open the window a stop() landing there needs.
  static void draft_test_poll_pause(int64_t us) {
    if constexpr (!Build::kFaults) {
      test_only("draft_test_poll_pause");
    } else {
      draft::g_test_poll_pause_us.store(us, std::memory_order_relaxed);
    }
  }

  // Layers made with any kernel's make_layer (kernel_layer), by id, for kernel_forward: the DSpark draft's CPU experts
  // (cpu_experts/draft.py) and tests.
  static std::mutex& kernel_layers_mutex() {
    static std::mutex mutex;
    return mutex;
  }
  // A test layer and the hidden size it was made with, which bounds kernel_forward's x and out.
  struct KernelLayer {
    cpu_experts::ExpertLayer layer;  // kernel null: dropped
    int64_t hidden = 0;
  };
  static std::vector<KernelLayer>& kernel_layers() {
    static std::vector<KernelLayer> layers;
    return layers;
  }
  static std::string& kernel_error_text() {
    static thread_local std::string text;
    return text;
  }

  // Kernel `kernel`'s make_layer over a slab table as set_cpu_layer takes it; returns the layer's id. The
  // kernel's std::invalid_argument propagates.
  static int64_t kernel_layer(int64_t kernel, TensorView slabs, int64_t capacity, int64_t hidden, int64_t intermediate,
                              int64_t activation, double act_limit, TensorView params) {
    const auto* k = reinterpret_cast<const cpu_experts::CpuExpertKernel*>(static_cast<intptr_t>(kernel));
    cpu_experts::ExpertLayer layer =
        k->make_layer(Base::layer_shape(slabs, capacity, hidden, intermediate, activation, act_limit),
                      Base::params_bytes(params));
    std::lock_guard<std::mutex> lock(kernel_layers_mutex());
    kernel_layers().push_back({layer, hidden});
    return static_cast<int64_t>(kernel_layers().size() - 1);
  }

  // One forward of layer `id` with its own kernel: x fp16 [rows, hidden] (every format's input today),
  // slots int32 and weights float32 [rows, k], out float32 [rows, hidden], hidden the layer's, all contiguous CPU;
  // cores int64 [n] (empty: unpinned). The shapes are checked against the layer, so a short x or out is refused here
  // rather than read or written past its end. Returns 0, 2 for std::invalid_argument, 1 for any other exception, its
  // message in kernel_error().
  static int64_t kernel_forward(int64_t id, TensorView x, TensorView slots, TensorView weights, TensorView out,
                                int64_t threads, TensorView cores, int64_t accumulate) {
    KernelLayer entry;
    {
      std::lock_guard<std::mutex> lock(kernel_layers_mutex());
      if (id < 0 || id >= static_cast<int64_t>(kernel_layers().size()) || !kernel_layers()[id].layer.kernel)
        throw std::runtime_error("kernel_forward: no layer " + std::to_string(id));
      entry = kernel_layers()[id];
    }
    using namespace host;
    auto cpu = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto k = SymbolicSize{"k"};
    const int64_t hidden = entry.hidden;
    // fp16 has no host-side dtype trait (fp16_t is CUDA-only), so the dtype is checked by hand.
    expert_stream::verify_named("x", TensorMatcher({rows, hidden}).with_device<kDLCPU>(cpu), x);
    if (x.dtype().code != kDLFloat || x.dtype().bits != 16 || x.dtype().lanes != 1)
      throw std::runtime_error("kernel_forward: x must be float16");
    expert_stream::verify_named("slots", TensorMatcher({rows, k}).with_dtype<int32_t>().with_device<kDLCPU>(cpu), slots);
    expert_stream::verify_named("weights", TensorMatcher({rows, k}).with_dtype<float>().with_device<kDLCPU>(cpu), weights);
    expert_stream::verify_named("out", TensorMatcher({rows, hidden}).with_dtype<float>().with_device<kDLCPU>(cpu), out);
    expert_stream::verify_named("cores", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), cores);
    const cpu_experts::ExpertLayer& layer = entry.layer;
    std::vector<int> on;
    const auto* c = static_cast<const int64_t*>(cores.data_ptr());
    for (int64_t i = 0; i < cores.size(0); ++i)
      on.push_back(static_cast<int>(c[i]));
    cpu_experts::ForwardCall call;
    call.rows = static_cast<int32_t>(slots.size(0));
    call.k = static_cast<int32_t>(slots.size(1));
    call.threads = static_cast<int32_t>(threads);
    call.x = x.data_ptr();
    call.slots = static_cast<const int32_t*>(slots.data_ptr());
    call.weights = static_cast<const float*>(weights.data_ptr());
    call.out = static_cast<float*>(out.data_ptr());
    call.accumulate = accumulate != 0;
    call.cores = on;
    kernel_error_text().clear();
    try {
      layer.kernel->check(layer, call);
      layer.kernel->forward(layer, call);
      return 0;
    } catch (const std::invalid_argument& e) {
      kernel_error_text() = e.what();
      return 2;
    } catch (const std::exception& e) {
      kernel_error_text() = e.what();
      return 1;
    }
  }

  static std::string kernel_error() {
    return kernel_error_text();
  }

  static void kernel_drop(int64_t id) {
    std::lock_guard<std::mutex> lock(kernel_layers_mutex());
    if (id >= 0 && id < static_cast<int64_t>(kernel_layers().size())) kernel_layers()[id] = {};
  }

  // Test only (HostCopyBackend): lets `marks` more copy marks complete (negative: all).
  static void copy_engine_release(int64_t handle, int64_t marks) {
    find(handle)->host_copy_backend().release(marks);
  }

  // Test only (HostCopyBackend): makes the issue and/or query calls fail.
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

  // Test only: a writer thread rewrites one record in a loop in the post kernel's seqlock order (seq = 0, fence,
  // payload, fence, a new seq) while this thread reads it with read_record. Every field of round r derives from r, so
  // a torn read shows. out = {records accepted, accepted records whose payload is not their seq's}.
  static void seqlock_stress(int64_t duration_ns, TensorView out) {
    if constexpr (!Build::kFaults) {
      test_only("seqlock_stress");
    } else {
      {
        using namespace host;
        auto cpu = SymbolicDevice{};
        expert_stream::verify_named("out", TensorMatcher({2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
      }
      alignas(128) uint8_t record[Wire::kRecordBytes] = {};
      std::atomic<bool> done{false};
      const auto count_of = [](uint32_t round) { return static_cast<uint16_t>(round % Wire::kLanes + 1); };
      // Every lane field of every cache line carries a value of the round, so a copy that mixes two rounds fails
      // `whole` whichever line it took from the other round.
      const auto id_of = [](uint32_t round, int j, int salt) {
        return static_cast<int16_t>(
            (round * 7u + static_cast<uint32_t>(j) * 131u + static_cast<uint32_t>(salt)) & 0x7FFFu);
      };
      const auto kind_of = [](uint32_t round, int j) { return static_cast<uint8_t>(1 + (round + j) % 5); };
      const auto weight_of = [](uint32_t round, int j) { return static_cast<float>((round & 0xFFFFu) + j); };
      std::thread writer([&] {
        for (uint32_t round = 1; !done.load(std::memory_order_relaxed); ++round) {
          const uint16_t row = static_cast<uint16_t>(round), protect = count_of(round);
          // All kLanes lanes are live; the protect ids vary in number.
          const uint8_t counts = Wire::kPackedCounts ? static_cast<uint8_t>(Wire::kLanes | protect << 4)
                                                     : static_cast<uint8_t>(Wire::kLanes);
          const uint8_t flags = static_cast<uint8_t>(round & 1u);
          const uint64_t chain = round;
          const uint32_t epoch = round * 3u;
          const int16_t id = static_cast<int16_t>(round & 0x7FFFu);
          store_release(record + Wire::kRecSeq, 0u);
          std::atomic_thread_fence(std::memory_order_seq_cst);
          std::memset(record + 4, 0, Wire::kRecordBytes - 4);
          std::memcpy(record + Wire::kRecRow, &row, 2);
          std::memcpy(record + Wire::kRecCounts, &counts, 1);
          std::memcpy(record + Wire::kRecFlags, &flags, 1);
          std::memcpy(record + Wire::kRecChain, &chain, 8);
          std::memcpy(record + Wire::kRecEpoch, &epoch, 4);
          if (!Wire::kPackedCounts) record[Wire::kRecProtectCount] = static_cast<uint8_t>(protect);
          for (int i = 0; i < protect; ++i)
            std::memcpy(record + Wire::kRecProtect + 2 * i, &id, 2);
          for (int j = 0; j < Wire::kLanes; ++j) {
            const int16_t expert = id_of(round, j, 1), slot = id_of(round, j, 2), dst = id_of(round, j, 3);
            const float weight = weight_of(round, j);
            const uint32_t kind_bits = static_cast<uint32_t>(kind_of(round, j)) << (4 * (j % 8));
            uint32_t word;
            std::memcpy(&word, record + Wire::kRecKinds + 4 * (j / 8), 4);
            word |= kind_bits;
            std::memcpy(record + Wire::kRecKinds + 4 * (j / 8), &word, 4);
            std::memcpy(record + Wire::kRecLaneExpert + 2 * j, &expert, 2);
            std::memcpy(record + Wire::kRecLaneSlot + 2 * j, &slot, 2);
            std::memcpy(record + Wire::kRecLaneDst + 2 * j, &dst, 2);
            std::memcpy(record + Wire::kRecLaneWeight + 4 * j, &weight, 4);
          }
          std::atomic_thread_fence(std::memory_order_seq_cst);
          store_release(record + Wire::kRecSeq, round * Wire::kDemandRecords + 1u);  // seqs of one ring slot
          // Hold every 64th record stable for a few us so a starved reader still gets a whole copy under CPU load.
          // A fixed spin count, not a deadline: this file reads the clock only where the census registers it.
          if (round % 64u == 0) {
            for (int spin = 0; spin < 256; ++spin)
              _mm_pause();
          }
        }
      });
      int64_t accepted = 0, torn = 0;
      const int64_t deadline = now_ns() + duration_ns;
      while (now_ns() < deadline) {
        const uint32_t seq = load_acquire(record + Wire::kRecSeq);
        Request request;
        if (seq == 0 || read_record(record, seq, &request) != RecordRead::kOk) continue;
        ++accepted;
        const uint32_t round = (seq - 1u) / Wire::kDemandRecords;
        bool whole = request.row == static_cast<uint16_t>(round) && request.captured == ((round & 1u) != 0) &&
                     request.protect.size() == count_of(round) && request.chain == round &&
                     request.gen == (static_cast<uint64_t>(round * 3u) << 32 | seq) &&
                     request.lanes.size() == static_cast<size_t>(Wire::kLanes);
        for (int32_t id : request.protect)
          whole = whole && id == static_cast<int16_t>(round & 0x7FFFu);
        for (size_t j = 0; whole && j < request.lanes.size(); ++j) {
          const Lane& lane = request.lanes[j];
          const int jj = static_cast<int>(j);
          whole = lane.expert == id_of(round, jj, 1) && lane.slot == id_of(round, jj, 2) &&
                  lane.dst == id_of(round, jj, 3) && lane.weight == weight_of(round, jj) &&
                  lane.kind == kind_of(round, jj);
        }
        if (!whole) ++torn;
      }
      done.store(true);
      writer.join();
      auto* result = static_cast<int64_t*>(out.data_ptr());
      result[0] = accepted;
      result[1] = torn;
    }
  }

  // Test only: read_record over one record (record: CPU uint8 [Wire::kRecordBytes]) as the service reads seq
  // `expected`.
  // out int64 [6 + Wire::kLanes + 1 + 5 * Wire::kLanes] = {status (RecordRead: 0 ok, 1 torn, 2 malformed), row,
  // captured, chain, gen, protect count, protect ids, lane count, then per lane: expert, slot, dst, kind, the weight's
  // bits}.
  static void read_record_fields(TensorView record, int64_t expected, TensorView out) {
    if constexpr (!Build::kFaults) {
      test_only("read_record_fields");
    } else {
      {
        using namespace host;
        auto cpu = SymbolicDevice{};
        expert_stream::verify_named(
            "record", TensorMatcher({Wire::kRecordBytes}).with_dtype<uint8_t>().with_device<kDLCPU>(cpu), record);
        expert_stream::verify_named(
            "out",
            TensorMatcher({6 + Wire::kLanes + 1 + 5 * Wire::kLanes}).with_dtype<int64_t>().with_device<kDLCPU>(cpu),
            out);
      }
      Request request;
      const RecordRead read =
          read_record(static_cast<const uint8_t*>(record.data_ptr()), static_cast<uint32_t>(expected), &request);
      auto* w = static_cast<int64_t*>(out.data_ptr());
      w[0] = static_cast<int64_t>(read);
      w[1] = request.row;
      w[2] = request.captured ? 1 : 0;
      w[3] = static_cast<int64_t>(request.chain);
      w[4] = static_cast<int64_t>(request.gen);
      w[5] = static_cast<int64_t>(request.protect.size());
      for (size_t i = 0; i < request.protect.size(); ++i)
        w[6 + i] = request.protect[i];
      w[6 + Wire::kLanes] = static_cast<int64_t>(request.lanes.size());
      for (size_t j = 0; j < request.lanes.size(); ++j) {
        const Lane& lane = request.lanes[j];
        int32_t bits;
        std::memcpy(&bits, &lane.weight, 4);
        int64_t* l = w + 7 + Wire::kLanes + 5 * j;
        l[0] = lane.expert;
        l[1] = lane.slot;
        l[2] = lane.dst;
        l[3] = lane.kind;
        l[4] = bits;
      }
    }
  }

  // Test only (HostCopyBackend): the marks closed so far, one per job issued.
  static int64_t copy_engine_marked(int64_t handle) {
    return find(handle)->host_copy_backend().marked();
  }

  // Test only (RamTier::inject): sleeps `delay_ns` before each demand read once `after_demands` demands have read
  // rows, and with `fail_reads` reports the reads as failed. InstrBuild only.
  static void inject(int64_t handle, int64_t delay_ns, int64_t fail_reads, int64_t after_demands) {
    if constexpr (!Build::kFaults) {
      test_only("inject");
    } else {
      find(handle)->inject(delay_ns, fail_reads != 0, after_demands);
    }
  }

  // Test only (RamTier::inject_group_stall): NUMA group `group`'s service sleeps `ns` before it reads its next record.
  // InstrBuild only.
  static void inject_group_stall(int64_t handle, int64_t group, int64_t ns) {
    if constexpr (!Build::kFaults) {
      test_only("inject_group_stall");
    } else {
      find(handle)->inject_group_stall(static_cast<int>(group), ns);
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

  // Test only: InstrBuild only (ProdBuild has no trace, so nothing to count). Gated on kMetrics, the trace's own flag.
  static int64_t trace_clock_reads() {
    if constexpr (!Build::kMetrics) {
      test_only("trace_clock_reads");
    } else {
      return expert_stream::traced_clock_reads().load(std::memory_order_relaxed);
    }
  }

  // Test only: the measured cost of one _mm_pause in ns, the service's idle-poll quantum (RamThread::run).
  static double pause_ns() {
    if constexpr (!Build::kFaults) {
      test_only("pause_ns");
    } else {
      constexpr int kProbe = 1 << 16;
      const int64_t start = now_ns();
      for (int i = 0; i < kProbe; ++i)
        _mm_pause();
      return static_cast<double>(now_ns() - start) / kProbe;
    }
  }
};

}  // namespace sglang::expert_stream

// The inner macro takes the HostTestExports type, so each line reads Exports::name like EXPERT_STREAM_HOST_EXPORTS'.
#define EXPERT_STREAM_HOST_TEST_EXPORTS(Exports) \
  EXPERT_STREAM_HOST_TEST_EXPORTS_OF(::sglang::expert_stream::HostTestExports<Exports>)
#define EXPERT_STREAM_HOST_TEST_EXPORTS_OF(Exports)                                                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_kernel_address, Exports::test_kernel_address);   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_kernel_calls, Exports::test_kernel_calls);       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_kernel_hold, Exports::test_kernel_hold);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_keep_warm_calls, Exports::test_keep_warm_calls); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_test_keep_warm_core, Exports::test_keep_warm_core);   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_test_post, Exports::draft_test_post);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_test_tear, Exports::draft_test_tear);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_test_finish_close, Exports::draft_test_finish_close); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_test_poll_pause, Exports::draft_test_poll_pause); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_kernel_layer, Exports::kernel_layer);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_kernel_forward, Exports::kernel_forward);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_kernel_error, Exports::kernel_error);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_kernel_drop, Exports::kernel_drop);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows, Exports::read_rows);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_traced, Exports::read_rows_traced);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_faulted, Exports::read_rows_faulted);       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_sqes, Exports::read_rows_sqes);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_publish_piece, Exports::publish_piece);               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_rows_pieces, Exports::read_rows_pieces);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_piece_geometry, Exports::piece_geometry);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pump, Exports::pump);                                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pump_group, Exports::pump_group);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_slot_info, Exports::slot_info);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_handled_through, Exports::handled_through);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_victim_census, Exports::victim_census);               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_busy_episode, Exports::busy_episode);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_idle, Exports::copy_engine_idle);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_release, Exports::copy_engine_release);   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_fail, Exports::copy_engine_fail);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_marked, Exports::copy_engine_marked);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_engine_ballast, Exports::copy_engine_ballast);   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_seqlock_stress, Exports::seqlock_stress);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_read_record_fields, Exports::read_record_fields);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject, Exports::inject);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject_fault, Exports::inject_fault);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject_group_stall, Exports::inject_group_stall);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_clock_reads, Exports::trace_clock_reads);       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pause_ns, Exports::pause_ns);
