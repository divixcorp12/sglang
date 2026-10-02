// The real host stack the bench drives: RamTier (its RowReader, the copy engine on its host backend, the CPU expert
// engine) and its RamThread, set up and torn down in the CPU-experts test's order
// (test_exl3_ram_miss_cpu_experts.py::_host). Production headers, unmodified.
#pragma once

#ifndef EXL3_FULL_STACK_INSTR
#error "EXL3_FULL_STACK_INSTR selects the host build: 0 ProdBuild, 1 InstrBuild"
#endif

#include <immintrin.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "aligned.h"
#include "device_sim.h"
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/build_policy.h"
#include "expert_stream/host/core_topology.h"
#include "expert_stream/host/row_reader.h"
#include "expert_stream/host/ram_thread.h"
#include "expert_stream/host/uring_reader.h"
#include "placement.h"
#include "row_images.h"

namespace fullstack {

namespace es = ::sglang::expert_stream;
using BenchBuild = std::conditional_t<EXL3_FULL_STACK_INSTR != 0, es::InstrBuild, es::ProdBuild>;
static_assert(kNames == static_cast<int>(::sglang::exl3::Exl3RowLayout::kNames.size()));

// exl3_ram_miss_host.cpp / exl3_ram_miss_host_instr.cpp's readers, minus the instrumented build's FaultyReader.
template <class Build>
struct ReaderFor;
template <>
struct ReaderFor<es::ProdBuild> {
  using type = es::UringReader;
};
template <>
struct ReaderFor<es::InstrBuild> {
  using type = es::InstrUringReader;
};

// exl3_ram_miss.py::_row_image_tables for one root: file `row` is the row's image, expert e's at e * row_stride,
// identity segments; then the reader's own image-table check.
inline es::Tables image_tables(const RowSet& set) {
  const ImageLayout& layout = set.layout;
  const auto rows = static_cast<int64_t>(set.paths.size());
  es::Tables t;
  t.layers = rows;
  t.experts = set.experts;
  t.parts = 1;
  t.slot_bytes = layout.image_bytes;
  t.paths = set.paths;
  t.source_paths = set.paths;
  t.file_sizes.assign(static_cast<size_t>(rows), set.experts * layout.row_stride);
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t e = 0; e < set.experts; ++e) t.extents.push_back(es::Read{row, e * layout.row_stride, layout.image_bytes, 0});
  t.starts.assign(static_cast<size_t>(rows * set.experts), 0);
  for (int n = 0; n < kNames; ++n) t.segments.push_back(es::Segment{n, 0, layout.name_offsets[n], layout.row_bytes[n]});
  t.need_end = layout.image_bytes;
  for (int64_t row = 0; row < rows; ++row) {
    t.slabs.emplace_back(set.slabs[row].begin(), set.slabs[row].end());
    for (int n = 0; n < kNames; ++n)
      t.buffer_regions.push_back(es::RegisteredRegion{set.slabs[row][n],
                                                      static_cast<size_t>(set.capacity * layout.row_bytes[n]),
                                                      static_cast<size_t>(layout.row_bytes[n])});
  }
  t.row_bytes.assign(layout.row_bytes.begin(), layout.row_bytes.end());
  t.images = true;
  es::check_image_tables<::sglang::exl3::Exl3RowLayout>(t);
  return t;
}

struct StackConfig {
  RowSet rows;
  int64_t staging = 3;
  es::CpuExpertForward forward = nullptr;
  int threads = 1;
  std::vector<int> cores;  // worker 0 first: the CPU expert thread pins itself there
  uint8_t* x_base = nullptr;
  int64_t x_stride = 0;
  uint8_t* out_base = nullptr;
  int64_t out_stride = 0;  // bytes; two parts of `hidden` floats per row
  int64_t hidden = 0;
  int service_cpu = -1;
  int copy_cpu = -1;
  int64_t wait_timeout_ns = 2'000'000'000;   // the watchdog's copy-wait deadline (SGLANG_DSV41_RAM_MISS_TIMEOUT_MS)
  int64_t fatal_wait_ns = 30'000'000'000;    // the watchdog's hung-request deadline
  std::array<int64_t, es::kLeaseLanes + 1> split{};
  size_t trace_capacity = 0;  // InstrBuild: the stage trace's ring, 0 off
};

template <class Build>
class Stack {
 public:
  using Source = es::RowReader<::sglang::exl3::Exl3RowLayout, typename ReaderFor<Build>::type, Build>;
  using Tier = es::RamTier<Source>;
  using Thread = es::RamThread<Tier>;
  static constexpr int64_t kCopySpinNs = 5'000'000;  // the transport's enable_copy_engine default (spin_us=5000)
  static constexpr int64_t kCpuSpinNs = 50'000'000;  // enable_cpu_experts' default (spin_us=50_000)

  // Construct on the writer's thread: the request page and lease block are first-touched here.
  explicit Stack(StackConfig config) : config_(std::move(config)) {
    rows_ = static_cast<int64_t>(config_.rows.paths.size());
    experts_ = config_.rows.experts;
    page_ = aligned_zeroed(es::kPageBytes);
    lease_bytes_ = es::kLeaseBlockBytes + round_up(rows_ * es::kDeltaStride, 4096);
    lease_ = aligned_zeroed(lease_bytes_);
    slot_map_.assign(static_cast<size_t>(rows_ * experts_), -1);
    tier_ = std::make_shared<Tier>(page_.get(), slot_map_.data(), lease_.get(), lease_bytes_, image_tables(config_.rows),
                                   std::vector<int64_t>(static_cast<size_t>(rows_), config_.rows.capacity),
                                   /*direct=*/true, /*hot_page=*/nullptr, 0);
    if (!tier_->open()) throw std::runtime_error("the tier's reader did not open (its error is on stderr)");
    tier_->reserve_staging(config_.staging);
    // The copy engine's thread and RamThread's watchdog inherit this thread's affinity: the copy CPU.
    PinScope copy(config_.copy_cpu);
    tier_->enable_copy_engine(-1, kCopySpinNs, config_.wait_timeout_ns);
    tier_->arm_copy_engine(true);
    es::CpuExpertConfig cpu;
    cpu.forward = config_.forward;
    cpu.x_base = config_.x_base;
    cpu.x_stride = config_.x_stride;
    cpu.out_base = config_.out_base;
    cpu.out_stride = config_.out_stride;
    cpu.out_part_stride = config_.hidden * static_cast<int64_t>(sizeof(float));
    cpu.hidden = config_.hidden;
    cpu.threads = config_.threads;
    cpu.cores = config_.cores;
    cpu.spin_ns = kCpuSpinNs;
    tier_->enable_cpu_experts(std::move(cpu), std::vector<int64_t>(config_.split.begin(), config_.split.end()));
    if constexpr (Build::kMetrics) {
      if (config_.trace_capacity > 0) tier_->enable_trace(config_.trace_capacity);
    }
    es::check_dedicated_core(config_.service_cpu, tier_->cpu_cores(), "full-stack bench: ");
    thread_ = std::make_unique<Thread>(tier_, config_.service_cpu, config_.fatal_wait_ns, /*spin_ns=*/0, /*busy_poll=*/true);
    thread_->start();
  }

  // The FFI's stop_thread order: open a gate left closed, stop the service, settle; the tier's destructor then stops
  // the copy and CPU expert threads.
  ~Stack() {
    if (tier_) tier_->open_closed_gate();
    if (thread_) thread_->stop();
    if (tier_) tier_->final_settle();
    thread_.reset();
    tier_.reset();
  }

  Stack(const Stack&) = delete;
  Stack& operator=(const Stack&) = delete;

  uint8_t* page() const {
    return page_.get();
  }
  uint8_t* lease() const {
    return lease_.get();
  }
  Tier& tier() {
    return *tier_;
  }
  const Tier& tier() const {
    return *tier_;
  }

  // The host's slot-map mirror (published after each read).
  int32_t mirror(int64_t row, int32_t expert) const {
    return __atomic_load_n(&slot_map_[static_cast<size_t>(row * experts_ + expert)], __ATOMIC_ACQUIRE);
  }

  bool wait_handled(uint32_t seq, int64_t deadline_ns) const {
    for (uint32_t spin = 0; !es::reached(tier_->handled_through(), seq); ++spin) {
      if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns) return false;
      _mm_pause();
    }
    return true;
  }

  void set_cpu_layer(int64_t row, int64_t handle) {
    tier_->set_cpu_layer(row, handle);
  }

  void set_split(const std::array<int64_t, es::kLeaseLanes + 1>& split) {
    tier_->set_cpu_split(split.data(), static_cast<int64_t>(split.size()));
  }

  std::array<int64_t, 3> cpu_stats() const {  // {jobs, lanes, forward ns}
    std::array<int64_t, 3> out{};
    tier_->cpu_stats(out.data());
    return out;
  }

  std::array<int64_t, es::kCounterCount> counters() const {
    std::array<int64_t, es::kCounterCount> out{};
    tier_->counters(out.data());
    return out;
  }

  // InstrBuild only (ProdBuild's drain_trace throws): discard every stage record so far. Unconstrained on purpose: the
  // callers' `if constexpr (BenchBuild::kMetrics)` sits in non-template functions, whose discarded branches are still
  // checked, so a constrained member would not compile in the prod build.
  void drain_all() {
    while (tier_->drain_trace(scratch_.get(), 1) == 1) {
    }
  }

  // InstrBuild: the next stage record, which must be request `seq`'s. The service pushes it as it finishes the request,
  // which can trail the CopyDone the writer saw: polled until deadline_ns.
  bool drain_stage(es::StageRecord& out, uint32_t seq, int64_t deadline_ns) {
    while (tier_->drain_trace(&out, 1) != 1) {
      if (monotonic_ns() > deadline_ns) return false;
      _mm_pause();
    }
    return out.seq == static_cast<int64_t>(seq);
  }

 private:
  StackConfig config_;
  int64_t rows_ = 0;
  int64_t experts_ = 0;
  // Declared before tier_ and thread_, so destroyed after them.
  AlignedBuffer page_;
  int64_t lease_bytes_ = 0;
  AlignedBuffer lease_;
  std::vector<int32_t> slot_map_;
  std::unique_ptr<es::StageRecord> scratch_ = std::make_unique<es::StageRecord>();
  std::shared_ptr<Tier> tier_;
  std::unique_ptr<Thread> thread_;
};

}  // namespace fullstack
