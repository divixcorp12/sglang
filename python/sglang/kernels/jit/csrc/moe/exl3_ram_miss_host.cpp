// Option C RAM-miss service for EXL3 streamed experts (DSV41 Phase 3b plan, D8-D19).
//
// This file grows in three plan tasks: the row reader (Task 10: io_uring superset
// reads into a page-aligned bounce, then Exl3ShardRowSource's per-name split into the
// pinned slabs), the C++-owned slot bookkeeping and request service (Task 11), and
// the service thread with its watchdog (Task 12). Nothing here makes a CUDA call:
// every write is a CPU store into (pinned) host memory (plan D9).


#include "expert_stream/host/ram_thread.h"

namespace sglang {
namespace expert_stream {

using tvm::ffi::TensorView;

}  // namespace expert_stream

using expert_stream::TensorView;

namespace {

std::vector<int64_t> slots_of(TensorView slots) {
  const auto* data = static_cast<const int64_t*>(slots.data_ptr());
  return std::vector<int64_t>(data, data + slots.size(0));
}

}  // namespace

// Read `experts` of streamed row `row` into `slots` once, synchronously (tests, tools).
// Arguments are validated by the Python wrapper (read_rows_once).
int64_t exl3_ram_miss_read_rows(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    std::string source_paths,
    int64_t slot_bytes,
    int64_t row_images,
    int64_t direct,
    int64_t row,
    TensorView experts,
    TensorView slots,
    int64_t step) {
  using namespace expert_stream;
  RowReader reader(tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images), direct != 0);
  if (!reader.open()) return 0;
  return reader.read(row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), [](size_t) { return false; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows, exl3_ram_miss_read_rows);

// Test only: exl3_ram_miss_read_rows with the reader's StageRecord copied to `record`
// (stage_words() int64), with `ok` and `status` set from the result. `fault` is the faulted call's
// tensor, laid out as exl3_ram_miss_read_rows_faulted's (kFaultWords words); an all-zero tensor injects nothing
// except that ordinal 0 selects row 0: the Python wrapper sends -1.
// `owner_core` (test-only owner-pinning scaffold, PACK_WORKERS.md): -1 (the Python wrapper's default)
// leaves the reader byte-for-byte what it is without this parameter; >= 0 pins the calling/owner thread
// to that core and excludes it from the packing pool's mask (RowReader::set_owner_core).
int64_t exl3_ram_miss_read_rows_traced(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
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
  using namespace expert_stream;
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  RowReader reader(
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images),
      direct != 0, f[19], f[20]);
  reader.set_owner_core(owner_core);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) return 0;
  reader.set_fault(fault_from(f));
  StageRecord stage;
  const int result = reader.read(
      row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), abandon_after(f[17]), &stage);
  stage.ok = result == 1 ? 1 : 0;
  stage.status = result == 1 ? kStatusServed : result == 0 ? kStatusFailed : kStatusCancelled;
  std::memcpy(record.data_ptr(), &stage, sizeof(stage));
  return result;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_traced, exl3_ram_miss_read_rows_traced);

// Test only: one reader reads `experts` into `slots` with `fault` injected
// (see ReadFault and fault_from), then reads `then_experts` into `then_slots` with no fault (no second read when
// there are none). Results go to
// `results[0..7]`: the two reads' results, the completions the reader had reaped after each, then its
// stale completions, generation wraps, the packing jobs still open when the first read returned and the
// number of packing workers the reader has.
void exl3_ram_miss_read_rows_faulted(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
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
  using namespace expert_stream;
  auto* out = static_cast<int64_t*>(results.data_ptr());
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  RowReader reader(
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images),
      direct != 0, f[19], f[20]);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) {
    out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = 0;
    return;
  }
  reader.set_fault(fault_from(f));
  const size_t step = f[18] > 0 ? static_cast<size_t>(f[18]) : static_cast<size_t>(kBounceRows);
  out[0] = reader.read(row, ids_of(experts), slots_of(slots), step, abandon_after(f[17]));
  out[2] = reader.cqes();
  out[4] = reader.stale_cqes();
  out[5] = reader.generation_wraps();
  out[6] = reader.unfinished_jobs();
  out[7] = reader.pack_workers();
  reader.set_fault(ReadFault{});
  if (then_experts.size(0) == 0) return;  // a test that only wants the first read's state
  out[1] = reader.read(row, ids_of(then_experts), slots_of(then_slots), kBounceRows, abandon_after(0));
  out[3] = reader.cqes();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_faulted, exl3_ram_miss_read_rows_faulted);

// Test only (U10): exl3_ram_miss_read_rows_traced's read, recording every SQE the reader prepared. `sqes` receives
// up to sqes.size(0) rows of 4 int64 (file, offset, length, bounce byte offset), in preparation order; `info` 5 int64:
// the result, the SQE count, the descriptor count, the ring credit and the completions reaped. `fault` as the faulted
// call's (word 22 turns piece streaming on).
void exl3_ram_miss_read_rows_sqes(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
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
  using namespace expert_stream;
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  auto* out = static_cast<int64_t*>(info.data_ptr());
  out[0] = out[1] = out[2] = out[3] = out[4] = 0;
  RowReader reader(
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images),
      direct != 0, f[19], f[20]);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) return;
  reader.set_fault(fault_from(f));
  std::vector<RowReader::SqeRecord> log;
  reader.set_sqe_log(&log);
  StageRecord stage;
  const int result = reader.read(
      row, ids_of(experts), slots_of(slots), static_cast<size_t>(step), abandon_after(f[17]), &stage);
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
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_sqes, exl3_ram_miss_read_rows_sqes);

// Test only (U8): the owner's publish primitive on one readiness word (`word`, one int64): 1 when it set `bit`.
int64_t exl3_ram_miss_publish_piece(TensorView word, int64_t generation, int64_t bit) {
  using namespace expert_stream;
  return publish_piece(
             static_cast<uint64_t*>(word.data_ptr()), static_cast<uint64_t>(generation), static_cast<uint8_t>(bit))
             ? 1
             : 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_publish_piece, exl3_ram_miss_publish_piece);

// Test only (U2, U3, U6): exl3_ram_miss_read_rows_traced's read, publishing each row's pieces into its readiness
// words: row ordinal o's are masks[o][0 .. masks.size(1)), under `generation` (the caller initialises them). When
// `reference` is not empty (a slab pointer table shaped like `slabs`, holding row o at ref_slots[o]), a checker
// thread polls the first word of every row while the read runs and, for each bit it sees set, compares the piece's
// bytes in the destination slab with the reference: what a device that acquired the bit would copy. `info` 5 int64:
// the result, the reader's refused publishes, the pieces checked, the pieces whose bytes differed, and the bits the
// checker saw set before the read returned.
void exl3_ram_miss_read_rows_pieces(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
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
  using namespace expert_stream;
  check_fault_words(fault);
  const auto* f = static_cast<const int64_t*>(fault.data_ptr());
  auto* out = static_cast<int64_t*>(info.data_ptr());
  std::fill(out, out + 5, 0);
  const Tables t = tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images);
  const std::vector<int32_t> ids = ids_of(experts);
  const std::vector<int64_t> dest = slots_of(slots);
  const size_t lanes = static_cast<size_t>(masks.size(1));
  if (static_cast<size_t>(masks.size(0)) != ids.size() || lanes == 0 || lanes > static_cast<size_t>(kPieceTargets)) {
    throw std::runtime_error("exl3 RAM miss: masks must be [rows, 1..8] readiness words");
  }
  auto* words = static_cast<uint64_t*>(masks.data_ptr());
  std::vector<PieceTarget> targets(ids.size());
  for (size_t o = 0; o < ids.size(); ++o) {
    for (size_t l = 0; l < lanes; ++l) targets[o].words[targets[o].count++] = words + o * lanes + l;
  }
  const PiecePublish publish{static_cast<uint64_t>(generation), targets.data()};
  RowReader reader(Tables(t), direct != 0, f[19], f[20]);
  if (f[22] != 0) reader.set_piece_stream(true);
  if (!reader.open()) return;
  reader.set_fault(fault_from(f));

  // The checker: the pieces' runs per row, then poll until the read returns, and once more after.
  const bool checking = reference.numel() > 0;
  const size_t count = t.segments.size();
  std::vector<PieceRun> runs(ids.size() * kPieces * count);
  const auto* ref_table = checking ? static_cast<const int64_t*>(reference.data_ptr()) : nullptr;
  const auto* ref_slot = checking ? static_cast<const int64_t*>(ref_slots.data_ptr()) : nullptr;
  if (checking) {
    for (size_t o = 0; o < ids.size(); ++o) {
      RowGeometry g;
      if (!row_geometry(t, static_cast<size_t>(row * t.experts + ids[o]), g, &runs[o * kPieces * count])) {
        throw std::runtime_error("exl3 RAM miss: the checker cannot cut a row");
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
          const auto* ref_base = reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(
              ref_table[row * static_cast<int64_t>(t.slabs[row].size()) + s.name]));
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
      while (reading.load(std::memory_order_acquire)) check_pass(true);
    });
  }
  StageRecord stage;
  int result = 0;
  try {
    result = reader.read(
        row, ids, dest, static_cast<size_t>(step), abandon_after(f[17]), &stage, nullptr, SIZE_MAX, nullptr, &publish);
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows_pieces, exl3_ram_miss_read_rows_pieces);

// Test only (U1): the sub-reads and pieces the reader computes when it admits expert `expert` of streamed row `row`
// (row_geometry). `subs`: kPieces rows of 6 int64 (file, offset, length, dest, part, k), in file order; `pieces`:
// kPieces rows of 1 + 2 * segments int64: the dependency mask, then (dst_lo, dst_hi) per segment in segment
// destination coordinates (dst + the run's bounds). Returns the sub-read count, or -1 when the row cannot be cut.
int64_t exl3_ram_miss_piece_geometry(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    std::string source_paths,
    int64_t slot_bytes,
    int64_t row_images,
    int64_t row,
    int64_t expert,
    TensorView subs,
    TensorView pieces) {
  using namespace expert_stream;
  const Tables t = tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images);
  const size_t count = t.segments.size();
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_piece_geometry, exl3_ram_miss_piece_geometry);

// The stream kernel's piece table (piece-streaming plan 4.2, open question 6: one entry per (row, expert), computed by
// the reader's own row_geometry so the device cannot disagree with it). `runs`: int32 [layers, experts, kPieces,
// segments, 2], each run as (dst_lo, dst_hi) byte offsets into the segment's name row. A row the reader refuses to
// cut gets empty runs; its read fails, so no device copy ever uses them. Returns how many rows were refused.
int64_t exl3_ram_miss_piece_runs(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    std::string source_paths,
    int64_t slot_bytes,
    int64_t row_images,
    TensorView runs) {
  using namespace expert_stream;
  const Tables t = tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images);
  const size_t count = t.segments.size();
  const size_t rows = static_cast<size_t>(t.layers * t.experts);
  if (runs.size(0) != t.layers || runs.size(1) != t.experts || runs.size(2) != kPieces ||
      runs.size(3) != static_cast<int64_t>(count) || runs.size(4) != 2) {
    throw std::runtime_error("exl3 RAM miss: the piece-run table has the wrong shape");
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
        throw std::runtime_error("exl3 RAM miss: a piece run ends past the int32 range of the stream kernel's table");
      }
      line[2 * k] = static_cast<int32_t>(segment.dst + piece[k].lo);
      line[2 * k + 1] = static_cast<int32_t>(segment.dst + piece[k].hi);
    }
  }
  return refused;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_piece_runs, exl3_ram_miss_piece_runs);

// Test only: build a packing pool as if the creating thread could run on the cores set in `inherited`
// (two int64 words, cores 0-127) and write each worker's affinity, as the kernel reports it, to `out`
// (two words per worker). Throws, like the pool, when no core is left.
void exl3_ram_miss_pack_pool_affinity(TensorView inherited, int64_t workers, TensorView out) {
  using namespace expert_stream;
  const auto* bits = static_cast<const int64_t*>(inherited.data_ptr());
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int core = 0; core < 128; ++core) {
    if ((static_cast<uint64_t>(bits[core / 64]) >> (core % 64)) & 1u) CPU_SET(core, &mask);
  }
  PackPool pool(static_cast<unsigned>(workers), mask, static_cast<size_t>(kBounceSlots));
  auto* words = static_cast<int64_t*>(out.data_ptr());
  for (size_t w = 0; w < pool.workers(); ++w) {
    const cpu_set_t set = pool.worker_affinity(w);
    words[2 * w] = words[2 * w + 1] = 0;
    for (int core = 0; core < 128; ++core) {
      if (CPU_ISSET(core, &set)) words[2 * w + core / 64] |= static_cast<int64_t>(uint64_t{1} << (core % 64));
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pack_pool_affinity, exl3_ram_miss_pack_pool_affinity);


// Test only: the cores a packing worker may use when the creating thread may use those set in `inherited`
// (two int64 words, cores 0-127), as two words in `out`. Starts no thread.
void exl3_ram_miss_pack_worker_cpus(TensorView inherited, TensorView out) {
  using namespace expert_stream;
  const auto* bits = static_cast<const int64_t*>(inherited.data_ptr());
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int core = 0; core < 128; ++core) {
    if ((static_cast<uint64_t>(bits[core / 64]) >> (core % 64)) & 1u) CPU_SET(core, &mask);
  }
  const cpu_set_t allowed = pack_worker_cpus(mask);
  auto* words = static_cast<int64_t*>(out.data_ptr());
  words[0] = words[1] = 0;
  for (int core = 0; core < 128; ++core) {
    if (CPU_ISSET(core, &allowed)) words[core / 64] |= static_cast<int64_t>(uint64_t{1} << (core % 64));
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pack_worker_cpus, exl3_ram_miss_pack_worker_cpus);

namespace expert_stream {
}  // namespace expert_stream

int64_t exl3_ram_miss_open(
    TensorView page,
    TensorView slot_map,
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    TensorView capacity,
    std::string paths,
    std::string source_paths,
    int64_t slot_bytes,
    int64_t row_images,
    int64_t direct,
    TensorView lease,
    int64_t pack_workers,
    TensorView hot_page) {
  using namespace expert_stream;
  const auto* capacity_data = static_cast<const int64_t*>(capacity.data_ptr());
  auto tier = std::make_shared<RamTier>(
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(slot_map.data_ptr()),
      static_cast<uint8_t*>(lease.data_ptr()),
      lease.size(0),
      tables_from(extents, starts, file_sizes, segments, slabs, row_bytes, paths, source_paths, slot_bytes, row_images),
      std::vector<int64_t>(capacity_data, capacity_data + capacity.size(0)),
      direct != 0,
      pack_workers,
      hot_page.size(0) ? static_cast<uint8_t*>(hot_page.data_ptr()) : nullptr,
      hot_page.size(0));
  if (!tier->open()) return -1;
  std::lock_guard<std::mutex> guard(registry_mutex());
  static int64_t next_handle = 1;
  const int64_t handle = next_handle++;
  registry().emplace(handle, std::move(tier));
  return handle;
}

// Defined after RamThread (the service thread block below).
void exl3_ram_miss_close(int64_t handle);

// 1 served a demand record, 3 a native-prefetch request, 2 an advisory record, 0 nothing posted. Refused while a
// thread pumps. The order is the service thread's: demand, prefetch, advisory.
int64_t exl3_ram_miss_pump(int64_t handle) {
  const auto tier = expert_stream::find(handle);
  if (tier->threaded()) throw std::runtime_error("exl3 RAM miss: pump() while the service thread runs");
  if (tier->pump_demand()) return 1;
  if (tier->pump_prefetch()) return 3;
  return tier->pump_advice() ? 2 : 0;
}

int64_t exl3_ram_miss_contains(int64_t handle, int64_t row, int64_t expert) {
  return expert_stream::find(handle)->has(row, expert) ? 1 : 0;
}

void exl3_ram_miss_touch(int64_t handle, int64_t row, int64_t expert) {
  expert_stream::find(handle)->touch(row, expert);
}

void exl3_ram_miss_assign(
    int64_t handle, int64_t row, int64_t expert, TensorView protect, int64_t fallback, TensorView out) {
  auto* result = static_cast<int64_t*>(out.data_ptr());
  int64_t evicted = -1;
  result[0] = expert_stream::find(handle)->assign(row, expert, expert_stream::ids_of(protect), fallback != 0, &evicted);
  result[1] = evicted;
}

// Prefill fills: `out` holds experts.size() + 1 int64, the claimed slots in order and then the evictions.
int64_t exl3_ram_miss_fill_begin(
    int64_t handle, int64_t row, TensorView experts, TensorView protect, int64_t fallback, TensorView out) {
  auto* result = static_cast<int64_t*>(out.data_ptr());
  const std::vector<int32_t> ids = expert_stream::ids_of(experts);
  return expert_stream::find(handle)->fill_begin(
      row, ids, expert_stream::ids_of(protect), fallback != 0, result, result + ids.size());
}

int64_t exl3_ram_miss_fill_wait(int64_t handle, int64_t rows, int64_t timeout_ns) {
  return expert_stream::find(handle)->fill_wait(rows, timeout_ns);
}

int64_t exl3_ram_miss_fill_landed(int64_t handle) {
  return expert_stream::find(handle)->fill_landed();
}

int64_t exl3_ram_miss_fill_end(int64_t handle) {
  return expert_stream::find(handle)->fill_end();
}

void exl3_ram_miss_release(int64_t handle, int64_t row, int64_t slot) {
  expert_stream::find(handle)->release(row, slot);
}

void exl3_ram_miss_slot_info(int64_t handle, int64_t row, TensorView out) {
  expert_stream::find(handle)->slot_info(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_lease_entry(int64_t handle, int64_t idx, TensorView out) {
  if (idx < 0 || idx >= expert_stream::kDemandRecords) throw std::runtime_error("exl3 RAM miss: request slot out of range");
  expert_stream::find(handle)->lease_entry(idx, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_inject_lease(int64_t handle, int64_t row, int64_t slot, int64_t delta) {
  expert_stream::find(handle)->inject_lease(row, slot, delta);
}

// out: free, evictable, leased.
void exl3_ram_miss_victim_census(int64_t handle, int64_t row, TensorView wanted, TensorView out) {
  const auto census = expert_stream::find(handle)->victim_census(row, expert_stream::ids_of(wanted));
  auto* result = static_cast<int64_t*>(out.data_ptr());
  result[0] = census.free;
  result[1] = census.evictable;
  result[2] = census.leased;
}

int64_t exl3_ram_miss_busy_since(int64_t handle) {
  return expert_stream::find(handle)->busy_since();
}

void exl3_ram_miss_close_admission(int64_t handle) {
  expert_stream::find(handle)->close_admission();
}

void exl3_ram_miss_set_lease_mode(int64_t handle, int64_t on) {
  expert_stream::find(handle)->set_lease_mode(on != 0);
}

void exl3_ram_miss_set_prefill_share(int64_t handle, int64_t share) {
  expert_stream::find(handle)->set_prefill_share(share);
}

void exl3_ram_miss_set_gpu_hot(int64_t handle, int64_t on) {
  expert_stream::find(handle)->set_gpu_hot(on != 0);
}

void exl3_ram_miss_set_two_phase(int64_t handle, int64_t on) {
  expert_stream::find(handle)->set_two_phase(on != 0);
}

void exl3_ram_miss_set_piece_stream(int64_t handle, int64_t on) {
  expert_stream::find(handle)->set_piece_stream(on != 0);
}

void exl3_ram_miss_enable_copy_engine(int64_t handle, int64_t device, int64_t spin_ns) {
  expert_stream::find(handle)->enable_copy_engine(device, spin_ns);
}

// entries: int64 [n, 3] of {source address, destination address, row bytes}; dst_rows: rows of every destination;
// sm_mask: the entries the copy wait reads itself (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES), 0 for none.
void exl3_ram_miss_set_copy_table(int64_t handle, int64_t row, TensorView entries, int64_t dst_rows, int64_t sm_mask) {
  if (entries.dim() != 2 || entries.size(1) != 3) throw std::runtime_error("exl3 RAM miss: copy table must be [n, 3]");
  expert_stream::find(handle)->set_copy_table(
      row, static_cast<const int64_t*>(entries.data_ptr()), entries.size(0), dst_rows, sm_mask);
}

void exl3_ram_miss_arm_copy_engine(int64_t handle, int64_t on) {
  expert_stream::find(handle)->arm_copy_engine(on != 0);
}

// page: pinned uint8 [kPrefetchPageBytes], the native-prefetch request and done lines.
void exl3_ram_miss_enable_native_prefetch(int64_t handle, TensorView page) {
  if (page.dim() != 1 || page.size(0) != expert_stream::kPrefetchPageBytes)
    throw std::runtime_error("exl3 RAM miss: the native prefetch page must be uint8 [256]");
  expert_stream::find(handle)->enable_native_prefetch(static_cast<uint8_t*>(page.data_ptr()));
}

// Test only: out int64 [3] = {active, row, slot} of the service's prefetch lease.
void exl3_ram_miss_prefetch_lease(int64_t handle, TensorView out) {
  expert_stream::find(handle)->prefetch_lease(static_cast<int64_t*>(out.data_ptr()));
}

int64_t exl3_ram_miss_copy_engine_idle(int64_t handle, int64_t timeout_ns) {
  return expert_stream::find(handle)->wait_copy_idle(expert_stream::now_ns() + timeout_ns) ? 1 : 0;
}

// Test only (HostCopyBackend): let `marks` more copy marks complete (negative: all), fail the calls, count marks.
void exl3_ram_miss_copy_engine_release(int64_t handle, int64_t marks) {
  expert_stream::find(handle)->host_copy_backend().release(marks);
}

void exl3_ram_miss_copy_engine_fail(int64_t handle, int64_t issue, int64_t query) {
  expert_stream::find(handle)->host_copy_backend().fail(issue != 0, query != 0);
}

// Test only: delay every copy job's completion by one extra copy of `bytes` from `src` to `dst` (0 bytes: off).
void exl3_ram_miss_copy_engine_ballast(int64_t handle, int64_t dst, int64_t src, int64_t bytes) {
  expert_stream::find(handle)->copy_engine_ballast(static_cast<uint64_t>(dst), static_cast<uint64_t>(src), bytes);
}

int64_t exl3_ram_miss_copy_engine_marked(int64_t handle) {
  return expert_stream::find(handle)->host_copy_backend().marked();
}

void exl3_ram_miss_inject_done_stall(int64_t handle, int64_t ns) {
  expert_stream::find(handle)->inject_done_stall(ns);
}

void exl3_ram_miss_mapping(int64_t handle, int64_t row, TensorView out) {
  expert_stream::find(handle)->mapping(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_slot_to_expert(int64_t handle, int64_t row, TensorView out) {
  expert_stream::find(handle)->slot_to_expert(row, static_cast<int64_t*>(out.data_ptr()));
}

int64_t exl3_ram_miss_lru_order(int64_t handle, int64_t row, TensorView out) {
  return expert_stream::find(handle)->lru_order(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_set_hot(int64_t handle, int64_t row, TensorView experts) {
  expert_stream::find(handle)->set_hot(row, static_cast<const int64_t*>(experts.data_ptr()), experts.size(0));
}

void exl3_ram_miss_inject(
    int64_t handle, int64_t delay_ns, int64_t fail_reads, int64_t after_demands, int64_t abandon_after_batches) {
  expert_stream::find(handle)->inject(delay_ns, fail_reads != 0, after_demands, abandon_after_batches);
}

// Test only: a full ReadFault for the tier's reader (the reader tests' fault tensor; see RamTier::inject_fault).
void exl3_ram_miss_inject_fault(int64_t handle, TensorView fault) {
  expert_stream::check_fault_words(fault);
  expert_stream::find(handle)->inject_fault(static_cast<const int64_t*>(fault.data_ptr()));
}

void exl3_ram_miss_counters(int64_t handle, TensorView out) {
  expert_stream::find(handle)->counters(static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_layer_rows(int64_t handle, int64_t advisory, TensorView out) {
  expert_stream::find(handle)->layer_rows(static_cast<int64_t*>(out.data_ptr()), advisory != 0);
}

int64_t exl3_ram_miss_trace_words() {
  return expert_stream::stage_words();
}

void exl3_ram_miss_trace_enable(int64_t handle, int64_t capacity) {
  if (capacity <= 0) throw std::runtime_error("exl3 RAM miss: the stage trace needs a positive capacity");
  expert_stream::find(handle)->enable_trace(static_cast<size_t>(capacity));
}

// Fills up to out.size(0) records, stage_words() int64 each; returns the count.
int64_t exl3_ram_miss_trace_drain(int64_t handle, TensorView out) {
  return expert_stream::find(handle)->drain_trace(
      static_cast<expert_stream::StageRecord*>(out.data_ptr()), out.size(0));
}

int64_t exl3_ram_miss_trace_clock_reads() {
  return expert_stream::traced_clock_reads().load(std::memory_order_relaxed);
}

int64_t exl3_ram_miss_trace_dropped(int64_t handle) {
  return expert_stream::find(handle)->trace_dropped();
}

// ---- Host-side simulated device: the post and wait kernels' protocol, for CPU tests ----

int64_t exl3_ram_miss_sim_post(
    TensorView page,
    int64_t row,
    TensorView need,
    TensorView protect,
    int64_t advisory,
    int64_t after,
    int64_t armed,
    int64_t lanes) {
  using namespace expert_stream;
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
int64_t exl3_ram_miss_sim_wait(TensorView page, int64_t seq, int64_t timeout_ns) {
  using namespace expert_stream;
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
void exl3_ram_miss_seqlock_stress(int64_t duration_ns, TensorView out) {
  using namespace expert_stream;
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_open, exl3_ram_miss_open);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_close, exl3_ram_miss_close);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pump, exl3_ram_miss_pump);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_contains, exl3_ram_miss_contains);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_touch, exl3_ram_miss_touch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_assign, exl3_ram_miss_assign);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_release, exl3_ram_miss_release);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_fill_begin, exl3_ram_miss_fill_begin);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_fill_wait, exl3_ram_miss_fill_wait);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_fill_landed, exl3_ram_miss_fill_landed);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_fill_end, exl3_ram_miss_fill_end);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_slot_info, exl3_ram_miss_slot_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject_lease, exl3_ram_miss_inject_lease);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_lease_entry, exl3_ram_miss_lease_entry);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_victim_census, exl3_ram_miss_victim_census);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_busy_since, exl3_ram_miss_busy_since);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_close_admission, exl3_ram_miss_close_admission);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_lease_mode, exl3_ram_miss_set_lease_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_gpu_hot, exl3_ram_miss_set_gpu_hot);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_prefill_share, exl3_ram_miss_set_prefill_share);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_two_phase, exl3_ram_miss_set_two_phase);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_piece_stream, exl3_ram_miss_set_piece_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_enable_copy_engine, exl3_ram_miss_enable_copy_engine);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_copy_table, exl3_ram_miss_set_copy_table);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_arm_copy_engine, exl3_ram_miss_arm_copy_engine);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_copy_engine_idle, exl3_ram_miss_copy_engine_idle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_enable_native_prefetch, exl3_ram_miss_enable_native_prefetch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_prefetch_lease, exl3_ram_miss_prefetch_lease);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_copy_engine_release, exl3_ram_miss_copy_engine_release);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_copy_engine_fail, exl3_ram_miss_copy_engine_fail);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_copy_engine_marked, exl3_ram_miss_copy_engine_marked);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_copy_engine_ballast, exl3_ram_miss_copy_engine_ballast);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject_done_stall, exl3_ram_miss_inject_done_stall);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_mapping, exl3_ram_miss_mapping);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_slot_to_expert, exl3_ram_miss_slot_to_expert);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_lru_order, exl3_ram_miss_lru_order);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_hot, exl3_ram_miss_set_hot);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject, exl3_ram_miss_inject);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject_fault, exl3_ram_miss_inject_fault);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_counters, exl3_ram_miss_counters);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_layer_rows, exl3_ram_miss_layer_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_words, exl3_ram_miss_trace_words);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_enable, exl3_ram_miss_trace_enable);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_drain, exl3_ram_miss_trace_drain);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_dropped, exl3_ram_miss_trace_dropped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_trace_clock_reads, exl3_ram_miss_trace_clock_reads);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_sim_post, exl3_ram_miss_sim_post);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_sim_wait, exl3_ram_miss_sim_wait);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_seqlock_stress, exl3_ram_miss_seqlock_stress);

namespace expert_stream {
}  // namespace expert_stream

void exl3_ram_miss_start_thread(int64_t handle, int64_t cpu_core, int64_t fatal_wait_ns, int64_t spin_ns) {
  using namespace expert_stream;
  if (cpu_core >= CPU_SETSIZE) throw std::runtime_error("exl3 RAM miss: cpu_core out of range");
  if (cpu_core >= 64 && cpu_core <= 71) {
    throw std::runtime_error("exl3 RAM miss: cores 64-71 are reserved (71 is production's doorbell core)");
  }
  if (cpu_core < 0) {
    cpu_set_t inherited;
    CPU_ZERO(&inherited);
    if (pthread_getaffinity_np(pthread_self(), sizeof(inherited), &inherited) == 0) {
      for (int core = 64; core <= 71; ++core) {
        if (CPU_ISSET(core, &inherited)) {
          std::fprintf(
              stderr,
              "WARNING exl3 RAM miss: the service thread inherits an affinity that includes reserved cores 64-71; "
              "run under taskset -c 0-63 or pass cpu_core\n");
          break;
        }
      }
    }
  }
  std::shared_ptr<RamTier> tier = find(handle);
  // Checked and registered under one lock, so a concurrent close() either sees the thread
  // (and joins it) or runs before it and leaves no handle to start it on.
  std::lock_guard<std::mutex> guard(registry_mutex());
  if (registry().count(handle) == 0) throw std::runtime_error("exl3 RAM miss: unknown handle");
  if (thread_registry().count(handle)) throw std::runtime_error("exl3 RAM miss: the service thread already runs");
  auto thread = std::make_shared<RamThread>(std::move(tier), static_cast<int>(cpu_core), fatal_wait_ns, spin_ns);
  thread->start();
  thread_registry()[handle] = std::move(thread);
}

void exl3_ram_miss_stop_thread(int64_t handle) {
  using namespace expert_stream;
  std::shared_ptr<RamThread> thread;
  {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto found = thread_registry().find(handle);
    if (found == thread_registry().end()) return;
    thread = std::move(found->second);
    thread_registry().erase(found);
  }
  thread->stop();
}

int64_t exl3_ram_miss_pause(int64_t handle, int64_t timeout_ns) {
  return expert_stream::find_thread(handle)->pause(timeout_ns);
}

void exl3_ram_miss_resume(int64_t handle) {
  expert_stream::find_thread(handle)->resume();
}

// Takes the tier and its service thread out of the registries under one lock (so no
// start_thread can slip in between), then joins the thread: it holds a reference to the
// tier, which writes through raw addresses of Python-owned tensors that the caller
// releases after this returns.
void exl3_ram_miss_close(int64_t handle) {
  using namespace expert_stream;
  std::shared_ptr<RamThread> thread;
  std::shared_ptr<RamTier> tier;
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_start_thread, exl3_ram_miss_start_thread);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_stop_thread, exl3_ram_miss_stop_thread);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pause, exl3_ram_miss_pause);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_resume, exl3_ram_miss_resume);

}  // namespace sglang
