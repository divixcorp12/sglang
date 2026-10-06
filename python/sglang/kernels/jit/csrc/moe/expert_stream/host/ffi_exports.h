// The expert-stream host exports: the FFI surface the server's path reaches, through expert_stream_transport.py.
//
// HostExports is written once for every row layout and file reader. An instantiation file names a layout, a reader
// and a build policy (build_policy.h) and expands EXPERT_STREAM_HOST_EXPORTS and EXPERT_STREAM_HOST_TEST_EXPORTS (the
// test and tool exports live in ffi_test_exports.h). See exl3/exl3_ram_miss_host.cpp (ProdBuild) and
// exl3/exl3_ram_miss_host_instr.cpp (InstrBuild).
//
// Every export takes an opaque `handle` naming a service in a per-instantiation registry; each call holds its own
// reference, so close() from another thread frees the service only after calls in flight return.
#pragma once

#include <sgl_kernel/tensor.h>

#include <map>
#include <memory>
#include <mutex>

#include "../tensor_checks.h"
#include "build_policy.h"
#include "core_topology.h"
#include "draft_cpu_thread.h"
#include "ram_thread.h"
#include "row_reader.h"

namespace sglang::expert_stream {

using tvm::ffi::TensorView;

/// \brief The host exports of one transport instantiation that the server's path reaches; HostTestExports adds the
/// test and tool exports the server does not call.
///
/// The function-local registries are per instantiation, and each layout is its own module, so one layout's handles can
/// never resolve in another's.
template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build>
struct HostExports {
  static_assert(BuildPolicy<Build>);
  using Source = RowReader<Layout, Reader, Build>;
  using Tier = RamTier<Source>;
  using Thread = RamThread<Tier>;

  // What the fault machinery compiles to in this build. The instantiation files static_assert these, so a ProdBuild
  // module that regained a fault entry, an SQE log or the ballast fails to compile.
  static constexpr bool kReaderFaults = requires(Source& reader, const ReadFault& fault) { reader.set_fault(fault); };
  static constexpr bool kSqeLog = requires(Source& reader) { reader.set_sqe_log(nullptr); };
  static constexpr bool kBallast = requires(Tier& tier) { tier.copy_engine_ballast(0, 0, 0); };

  // Checks the table tensors every reader entry takes, once here rather than per read. tables_from dereferences them
  // through raw pointers with no dtype or device check of its own, so this runs before it: a wrong-dtype or too-narrow
  // table raises here, naming the tensor.
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
    verify_named(
        "buffer_regions", TensorMatcher({-1, 3}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), buffer_regions);
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

  // The wire this module was compiled for (-DSGLANG_EXPERT_STREAM_LANES / _NODES), for the Python side's checks.
  static int64_t wire_lanes() {
    return Wire::kLanes;
  }
  static int64_t wire_nodes() {
    return Wire::kNodes;
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

  // Fills the stream kernel's piece table: one entry per (row, expert), computed by the reader's own row_geometry so
  // the device cannot disagree with it. `runs`: int32 [layers, experts, kPieces, segments, 2], each run as
  // (dst_lo, dst_hi) byte offsets into the segment's name row. A row the reader refuses to cut gets empty runs; its
  // read fails, so no device copy ever uses them. Returns how many rows were refused.
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
      TensorView hot_page,
      TensorView ranges,
      TensorView sq_thread_cpus) {
    using namespace host;
    // Records and map deltas carry expert ids and slots as i16 (lease_layout.h).
    RuntimeCheck(
        starts.dim() == 2 && starts.size(1) <= Wire::kRecIdMax, "starts: records carry expert ids up to ", Wire::kRecIdMax);
    auto capacity_mem = SymbolicDevice{};
    verify_named(
        "capacity", TensorMatcher({starts.size(0)}).with_dtype<int64_t>().with_device<kDLCPU>(capacity_mem), capacity);
    for (int64_t row = 0; row < capacity.size(0); ++row)
      RuntimeCheck(
          static_cast<const int64_t*>(capacity.data_ptr())[row] <= Wire::kRecIdMax,
          "capacity: records carry slots up to ",
          Wire::kRecIdMax);
    check_table_tensors(extents, starts, file_sizes, segments, slabs, row_bytes, buffer_regions);
    // page, slot_map and lease are pinned (or not) together (ExpertStreamHost.__init__), so one SymbolicDevice
    // ties them to the same actual device; capacity is always a plain CPU tensor. hot_page is optional (an
    // absent one is an unpinned torch.empty(0)) and gets its own SymbolicDevice so it is not forced to equal
    // page's device when it is not given.
    auto host_mem = SymbolicDevice{};
    verify_named(
        "page", TensorMatcher({Wire::kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(host_mem), page);
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
    // Each group's slot range of every row, as [lo, hi), and its ring's SQPOLL core.
    verify_named(
        "ranges",
        TensorMatcher({Wire::kNodes, extents.size(0), 2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu),
        ranges);
    verify_named(
        "sq_thread_cpus", TensorMatcher({Wire::kNodes}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), sq_thread_cpus);
    const auto* capacity_data = static_cast<const int64_t*>(capacity.data_ptr());
    const auto* range_data = static_cast<const int64_t*>(ranges.data_ptr());
    std::vector<std::vector<std::pair<int64_t, int64_t>>> group_ranges(Wire::kNodes);
    for (int g = 0; g < Wire::kNodes; ++g)
      for (int64_t row = 0; row < extents.size(0); ++row) {
        const int64_t* range = range_data + (g * extents.size(0) + row) * 2;
        group_ranges[g].emplace_back(range[0], range[1]);
      }
    const auto* sq_data = static_cast<const int64_t*>(sq_thread_cpus.data_ptr());
    std::vector<int> sq_cpus;
    for (int g = 0; g < Wire::kNodes; ++g)
      sq_cpus.push_back(static_cast<int>(sq_data[g]));
    auto tier = std::make_shared<RamTier<Source>>(
        static_cast<uint8_t*>(page.data_ptr()),
        static_cast<int32_t*>(slot_map.data_ptr()),
        static_cast<uint8_t*>(lease.data_ptr()),
        lease.size(0),
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
        std::vector<int64_t>(capacity_data, capacity_data + capacity.size(0)),
        direct != 0,
        hot_page.size(0) ? static_cast<uint8_t*>(hot_page.data_ptr()) : nullptr,
        hot_page.size(0),
        std::move(group_ranges),
        std::move(sq_cpus));
    if (!tier->open()) return -1;
    std::lock_guard<std::mutex> guard(registry_mutex());
    static int64_t next_handle = 1;
    const int64_t handle = next_handle++;
    registry().emplace(handle, std::move(tier));
    return handle;
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

  static void close_admission(int64_t handle) {
    find(handle)->close_admission();
  }

  static void set_prefill_share(int64_t handle, int64_t share) {
    find(handle)->set_prefill_share(share);
  }

  // `cpus` is int64 [n], the copy thread's affinity (ThreadingConfig.copy_cpus); empty inherits the caller's.
  // `spin_ns`: how long the idle copy thread polls before it sleeps on its doorbell; < 0 never sleeps.
  static void enable_copy_engine(
      int64_t handle, int64_t device, int64_t spin_ns, int64_t wait_timeout_ns, TensorView cpus) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("cpus", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), cpus);
    std::vector<int> list;
    const auto* c = static_cast<const int64_t*>(cpus.data_ptr());
    for (int64_t i = 0; i < cpus.size(0); ++i)
      list.push_back(static_cast<int>(c[i]));
    find(handle)->enable_copy_engine(device, spin_ns, wait_timeout_ns, std::move(list));
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

  // A slab table and a layer's scalars as the ExpertLayer make_layer takes (set_cpu_layer, the test export kernel_layer):
  // `slabs` int64 [n, 2] of {address, slot bytes} in the format's slab order (address 0: an absent optional slab),
  // n <= kMaxSlabs.
  static cpu_experts::ExpertLayer layer_shape(
      TensorView slabs, int64_t capacity, int64_t hidden, int64_t intermediate, int64_t activation, double act_limit) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("slabs", TensorMatcher({-1, 2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), slabs);
    if (slabs.size(0) > cpu_experts::kMaxSlabs)
      throw std::runtime_error(error_prefix<Layout>() + "a CPU expert layer has at most " +
                               std::to_string(cpu_experts::kMaxSlabs) + " slabs");
    cpu_experts::ExpertLayer d;
    d.capacity = static_cast<int32_t>(capacity);
    d.hidden = static_cast<int32_t>(hidden);
    d.intermediate = static_cast<int32_t>(intermediate);
    d.activation = static_cast<int32_t>(activation);
    d.act_limit = static_cast<float>(act_limit);
    d.slab_count = static_cast<int32_t>(slabs.size(0));
    const auto* s = static_cast<const int64_t*>(slabs.data_ptr());
    for (int i = 0; i < d.slab_count; ++i) {
      d.slabs[i] = reinterpret_cast<const void*>(static_cast<intptr_t>(s[2 * i]));
      d.slot_bytes[i] = static_cast<uint64_t>(s[2 * i + 1]);
    }
    return d;
  }

  // A layer's params tensor (uint8 [m], the format's parameter struct) as make_layer's bytes; empty for m = 0.
  static std::span<const std::byte> params_bytes(TensorView params) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("params", TensorMatcher({-1}).with_dtype<uint8_t>().with_device<kDLCPU>(cpu), params);
    return {static_cast<const std::byte*>(params.data_ptr()), static_cast<size_t>(params.size(0))};
  }

  // Enables NUMA group `group`'s CPU experts. `kernel` is the format's CpuExpertKernel's address (the trait's
  // kernel_address()), which must outlive the service; rows join once their layer is made (set_cpu_layer). `split` is
  // int64 [Wire::kLanes + 1], CPU lanes per n eligible lanes;
  // `cores` is int64 [n], the CPU expert thread's affinity (may be empty). `x_rows` is uint8 [rows, stride] in host
  // memory, where the post kernel writes a row's input; `out_rows` is float32 [rows, >= Wire::kNodes * parts * hidden]
  // in host memory, where the device reads a row's CPU partial sums (group g's part 0 the CPU hits', part 1 the CPU
  // misses' when parts is 2, at parts 2g and 2g + 1). Both tensors must outlive the service. The idle engine holds its
  // team in the kernel's keep_warm, in register work for `keep_warm_ns` after each job and in PAUSE for `spin_ns` after
  // that, then sleeps until the next submit; `spin_ns` < 0 holds the team until the next submit, however long.
  static void enable_cpu_experts(
      int64_t handle,
      int64_t group,
      int64_t kernel,
      TensorView split,
      TensorView cores,
      TensorView x_rows,
      TensorView out_rows,
      int64_t hidden,
      int64_t parts,
      int64_t threads,
      int64_t spin_ns,
      int64_t keep_warm_ns) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    auto host_mem = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    expert_stream::verify_named(
        "split", TensorMatcher({expert_stream::Wire::kLanes + 1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), split);
    expert_stream::verify_named("cores", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), cores);
    expert_stream::verify_named(
        "x_rows", TensorMatcher({rows, -1}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(host_mem), x_rows);
    expert_stream::verify_named(
        "out_rows", TensorMatcher({rows, -1}).with_dtype<float>().with_device<kDLCPU, kDLCUDAHost>(host_mem), out_rows);
    if (kernel == 0) throw std::runtime_error(error_prefix<Layout>() + "CPU experts need the format's kernel");
    expert_stream::CpuExpertConfig config;
    config.kernel = reinterpret_cast<const cpu_experts::CpuExpertKernel*>(static_cast<intptr_t>(kernel));
    const auto* c = static_cast<const int64_t*>(cores.data_ptr());
    for (int64_t i = 0; i < cores.size(0); ++i)
      config.cores.push_back(static_cast<int>(c[i]));
    config.x_base = static_cast<const uint8_t*>(x_rows.data_ptr());
    config.x_stride = x_rows.size(1);
    if (parts != 1 && parts != 2)
      throw std::runtime_error(error_prefix<Layout>() + "CPU experts write 1 or 2 output parts");
    if (group < 0 || group >= expert_stream::Wire::kNodes)
      throw std::runtime_error(error_prefix<Layout>() + "CPU experts name a group the build does not have");
    if (out_rows.size(1) < expert_stream::Wire::kNodes * parts * hidden)
      throw std::runtime_error(error_prefix<Layout>() + "out_rows is narrower than every group's parts");
    config.out_base = static_cast<uint8_t*>(out_rows.data_ptr()) + group * parts * hidden * sizeof(float);
    config.out_stride = out_rows.size(1) * static_cast<int64_t>(sizeof(float));
    config.out_part_stride = parts == 2 ? hidden * static_cast<int64_t>(sizeof(float)) : 0;
    config.hidden = hidden;
    config.threads = static_cast<int>(threads);
    config.spin_ns = spin_ns;
    if (keep_warm_ns < 0) throw std::runtime_error(error_prefix<Layout>() + "the keep-warm window is negative");
    config.keep_warm_ns = keep_warm_ns;
    const auto* sp = static_cast<const int64_t*>(split.data_ptr());
    find(handle)->enable_cpu_experts(
        static_cast<int>(group), std::move(config), std::vector<int64_t>(sp, sp + split.size(0)));
  }

  // CPU experts: `row`'s layer, made here by the enabled kernel's make_layer from the row's pinned slabs. `slabs` is
  // int64 [n, 2] of {address, slot bytes} in the format's slab order (address 0: an absent optional slab), `params`
  // uint8 [bytes] the format's parameter struct. Once per row, at any time; the slabs must outlive the service.
  static void set_cpu_layer(
      int64_t handle,
      int64_t row,
      TensorView slabs,
      int64_t capacity,
      int64_t hidden,
      int64_t intermediate,
      int64_t activation,
      double act_limit,
      TensorView params) {
    find(handle)->make_cpu_layer(
        row, layer_shape(slabs, capacity, hidden, intermediate, activation, act_limit), params_bytes(params));
  }

  // The DSpark draft's CPU threads (draft_cpu_thread.h), by handle. Shared so a stop on one thread and stats on another
  // never free a thread under a call.
  static std::mutex& draft_mutex() {
    static std::mutex mutex;
    return mutex;
  }
  static std::map<int64_t, std::shared_ptr<draft::DraftCpuThread>>& draft_registry() {
    static std::map<int64_t, std::shared_ptr<draft::DraftCpuThread>> threads;
    return threads;
  }
  static std::shared_ptr<draft::DraftCpuThread> find_draft(int64_t handle) {
    std::lock_guard<std::mutex> guard(draft_mutex());
    const auto found = draft_registry().find(handle);
    if (found == draft_registry().end()) throw std::runtime_error("DSpark draft CPU experts: no thread " + std::to_string(handle));
    return found->second;
  }

  // Opens a draft CPU thread over a DraftCpuAreas (dspark_draft_cpu.py): `channel` uint8 [kChannelBytes], 128-byte
  // aligned; `x` fp16 [stages, kMaxRows, hidden]; `slots` int32 and `weights` fp32 [stages, kMaxRows, kMaxK]; `out` fp32
  // [stages, kMaxRows, hidden], all in host memory and alive until draft_cpu_stop. `cores` int64 [n] as
  // enable_cpu_experts' (check_cpu_expert_team). Idle as enable_cpu_experts' engine; `fatal_wait_ns` bounds how long a
  // posted record may stay incomplete.
  static int64_t draft_cpu_open(
      TensorView channel,
      TensorView x,
      TensorView slots,
      TensorView weights,
      TensorView out,
      int64_t hidden,
      int64_t stages,
      int64_t threads,
      TensorView cores,
      int64_t spin_ns,
      int64_t keep_warm_ns,
      int64_t fatal_wait_ns) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    auto host_mem = SymbolicDevice{};
    const int64_t rows = draft::kMaxRows, k = draft::kMaxK;
    expert_stream::verify_named(
        "channel",
        TensorMatcher({draft::kChannelBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(host_mem),
        channel);
    // fp16 has no host-side dtype trait (fp16_t is CUDA-only), so the dtype is checked by hand.
    expert_stream::verify_named("x", TensorMatcher({stages, rows, hidden}).with_device<kDLCPU, kDLCUDAHost>(host_mem), x);
    if (x.dtype().code != kDLFloat || x.dtype().bits != 16 || x.dtype().lanes != 1)
      throw std::runtime_error("DSpark draft CPU experts: x must be float16");
    expert_stream::verify_named(
        "slots", TensorMatcher({stages, rows, k}).with_dtype<int32_t>().with_device<kDLCPU, kDLCUDAHost>(host_mem), slots);
    expert_stream::verify_named(
        "weights", TensorMatcher({stages, rows, k}).with_dtype<float>().with_device<kDLCPU, kDLCUDAHost>(host_mem), weights);
    expert_stream::verify_named(
        "out", TensorMatcher({stages, rows, hidden}).with_dtype<float>().with_device<kDLCPU, kDLCUDAHost>(host_mem), out);
    expert_stream::verify_named("cores", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), cores);
    if (reinterpret_cast<uintptr_t>(channel.data_ptr()) % 128 != 0)
      throw std::runtime_error("DSpark draft CPU experts: the channel is not 128-byte aligned");
    draft::DraftCpuThread::Config config;
    config.channel = static_cast<uint8_t*>(channel.data_ptr());
    config.x = static_cast<const uint8_t*>(x.data_ptr());
    config.slots = static_cast<const int32_t*>(slots.data_ptr());
    config.weights = static_cast<const float*>(weights.data_ptr());
    config.out = static_cast<float*>(out.data_ptr());
    config.hidden = hidden;
    config.stages = static_cast<int>(stages);
    config.threads = static_cast<int>(threads);
    const auto* c = static_cast<const int64_t*>(cores.data_ptr());
    for (int64_t i = 0; i < cores.size(0); ++i)
      config.cores.push_back(static_cast<int>(c[i]));
    config.spin_ns = spin_ns;
    config.keep_warm_ns = keep_warm_ns;
    config.fatal_wait_ns = fatal_wait_ns;
    config.test_hooks = Build::kFaults;
    auto thread = std::make_shared<draft::DraftCpuThread>(std::move(config));
    static std::atomic<int64_t> next_handle{1};
    const int64_t handle = next_handle.fetch_add(1);
    std::lock_guard<std::mutex> guard(draft_mutex());
    draft_registry()[handle] = std::move(thread);
    return handle;
  }

  // `stage`'s layer, made by `kernel`'s make_layer (the arguments as set_cpu_layer's); before draft_cpu_start.
  static void draft_cpu_set_layer(
      int64_t handle,
      int64_t stage,
      int64_t kernel,
      TensorView slabs,
      int64_t capacity,
      int64_t hidden,
      int64_t intermediate,
      int64_t activation,
      double act_limit,
      TensorView params) {
    if (kernel == 0) throw std::runtime_error("DSpark draft CPU experts need the format's kernel");
    const auto* k = reinterpret_cast<const cpu_experts::CpuExpertKernel*>(static_cast<intptr_t>(kernel));
    find_draft(handle)->set_layer(
        static_cast<int>(stage),
        k->make_layer(layer_shape(slabs, capacity, hidden, intermediate, activation, act_limit), params_bytes(params)));
  }

  static void draft_cpu_start(int64_t handle) {
    find_draft(handle)->start();
  }

  // Stops the thread (draft_cpu_thread.h's stop) and forgets the handle; a second stop does nothing.
  static void draft_cpu_stop(int64_t handle) {
    std::shared_ptr<draft::DraftCpuThread> thread;
    {
      std::lock_guard<std::mutex> guard(draft_mutex());
      const auto found = draft_registry().find(handle);
      if (found == draft_registry().end()) return;
      thread = std::move(found->second);
      draft_registry().erase(found);
    }
    thread->stop();
  }

  // int64 [7]: jobs, rows, forward ns, keep-warm holds, collided jobs, shared routes, collided forward ns.
  static void draft_cpu_stats(int64_t handle, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({7}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const std::shared_ptr<draft::DraftCpuThread> thread = find_draft(handle);
    auto* o = static_cast<int64_t*>(out.data_ptr());
    o[0] = thread->jobs();
    o[1] = thread->rows();
    o[2] = thread->forward_ns();
    o[3] = thread->holds();
    o[4] = thread->collided_jobs();
    o[5] = thread->shared_routes();
    o[6] = thread->collided_forward_ns();
  }

  // Reserves every row's staging slots (up to k, fewer on a small tier) and publishes its first map delta
  // (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Deltas and the bulk delta"). Call once, paused or before the thread
  // starts, with the tier empty.
  static void reserve_staging(int64_t handle, int64_t k) {
    find(handle)->reserve_staging(k);
  }

  // The eager paths' map changes since the last call: bulk_delta_count, then take_bulk_delta into int32 [that, 3] of
  // {row, expert, slot}. A paused caller only; the count call joins a running fill first, so its unmaps are counted.
  static int64_t bulk_delta_count(int64_t handle) {
    return find(handle)->bulk_delta_count();
  }

  static void take_bulk_delta(int64_t handle, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1, 3}).with_dtype<int32_t>().with_device<kDLCPU>(cpu), out);
    find(handle)->take_bulk_delta(static_cast<int32_t*>(out.data_ptr()), out.size(0));
  }

  // CPU experts: installs group `group`'s new split table (int64 [Wire::kLanes + 1]), at any time.
  static void set_cpu_split(int64_t handle, int64_t group, TensorView split) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("split", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), split);
    find(handle)->set_cpu_split(static_cast<int>(group), static_cast<const int64_t*>(split.data_ptr()), split.size(0));
  }

  // Group `group`'s CPU experts' metrics: out int64 [3] = {jobs, lanes, forward ns}.
  static void cpu_stats(int64_t handle, int64_t group, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({3}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    find(handle)->cpu_stats(static_cast<int>(group), static_cast<int64_t*>(out.data_ptr()));
  }

  // CPU experts' calibration: the bytes the DMA moves per expert of `row`.
  static int64_t copy_expert_bytes(int64_t handle, int64_t row) {
    return find(handle)->copy_expert_bytes(row);
  }

  // CPU experts' startup calibration (split_calibration.h): out float64 [kCalibRows, kCalibCols] ms. The caller owns the tier.
  static void calibrate_cpu_split(
      int64_t handle,
      int64_t group,
      int64_t row,
      int64_t device,
      int64_t reps,
      int64_t scratch,
      int64_t scratch_bytes,
      int64_t timeout_ns,
      TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named(
        "out",
        TensorMatcher({expert_stream::kCalibRows, expert_stream::kCalibCols})
            .with_dtype<double>()
            .with_device<kDLCPU>(cpu),
        out);
    find(handle)->calibrate_cpu_split(
        static_cast<int>(group),
        row,
        device,
        reps,
        static_cast<uint64_t>(scratch),
        scratch_bytes,
        timeout_ns,
        static_cast<double*>(out.data_ptr()));
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

  // Group `group`'s own counters: its service thread's block (RamTier::group_counters).
  static void group_counters(int64_t handle, int64_t group, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named(
        "out", TensorMatcher({expert_stream::kCounterCount}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    if (group < 0 || group >= tier->groups())
      throw std::runtime_error(error_prefix<Layout>() + "group " + std::to_string(group) + " is out of range");
    tier->group_counters(static_cast<int>(group), static_cast<int64_t*>(out.data_ptr()));
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

  static void layer_rows(int64_t handle, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const auto tier = find(handle);
    // Exact: RamTier::layer_rows writes one word per streamed layer through a raw pointer with no bound.
    expert_stream::verify_named(
        "out", TensorMatcher({tier->layers()}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    tier->layer_rows(static_cast<int64_t*>(out.data_ptr()));
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

  static int64_t trace_dropped(int64_t handle) {
    return find(handle)->trace_dropped();
  }

  // `cpu_cores`: one core per NUMA group, -1 inherits the caller's affinity. The reserved-core rule is
  // ExpertStreamHost.start_thread's (check_not_reserved). `spin_ns`: how long an idle service thread polls before it
  // sleeps between polls; < 0 never sleeps.
  static void start_thread(
      int64_t handle, TensorView cpu_cores, int64_t fatal_wait_ns, int64_t spin_ns, int64_t busy_poll) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named(
        "cpu_cores", TensorMatcher({Wire::kNodes}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), cpu_cores);
    const auto* core_data = static_cast<const int64_t*>(cpu_cores.data_ptr());
    std::vector<int> cores;
    bool inherits = false;
    for (int g = 0; g < Wire::kNodes; ++g) {
      if (core_data[g] >= CPU_SETSIZE) throw std::runtime_error(error_prefix<Layout>() + "cpu_core out of range");
      if (core_data[g] >= 64 && core_data[g] <= 71) {
        throw std::runtime_error(
            error_prefix<Layout>() + "cores 64-71 are reserved (NVMe completion interrupts are pinned there)");
      }
      inherits = inherits || core_data[g] < 0;
      cores.push_back(static_cast<int>(core_data[g]));
    }
    if (inherits) {
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
    if (busy_poll != 0) {
      for (const int core : cores)
        check_dedicated_core(core, tier->cpu_cores(), error_prefix<Layout>());
    }
    // Checked and registered under one lock, so a concurrent close() either sees the thread
    // (and joins it) or runs before it and leaves no handle to start it on.
    std::lock_guard<std::mutex> guard(registry_mutex());
    if (registry().count(handle) == 0) throw std::runtime_error(error_prefix<Layout>() + "unknown handle");
    if (thread_registry().count(handle))
      throw std::runtime_error(error_prefix<Layout>() + "the service thread already runs");
    auto thread = std::make_shared<Thread>(std::move(tier), std::move(cores), fatal_wait_ns, spin_ns, busy_poll != 0);
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
    // Before the join: an in-flight replay's copy wait would otherwise wait for a copy thread that is gone.
    if (tier) tier->open_closed_gate();
    thread->stop();
    // The final settle: joins a prefill fill still running after a stop mid-pause and runs its epilogue, before
    // ExpertStreamHost.stop writes its counters line. Here and not in RamThread::stop, which ~RamThread also runs and
    // which could then touch a page the Python side already freed; at this point the thread has joined and the page
    // is still alive, because close() runs after this call.
    if (tier) tier->final_settle();
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
    // A service still running also ends any armed copy wait first (its lease block is alive: the thread uses it).
    if (thread && tier) tier->open_closed_gate();
    if (thread) thread->stop();
  }
};

}  // namespace sglang::expert_stream

// One line per export; this list and ffi_test_exports.h's are the module's whole Python-visible surface.
#define EXPERT_STREAM_HOST_EXPORTS(Exports)                                                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_build_name, Exports::build_name);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_wire_lanes, Exports::wire_lanes);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_wire_nodes, Exports::wire_nodes);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_layout_names, Exports::layout_names);               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_layout_small_mask, Exports::layout_small_mask);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_piece_runs, Exports::piece_runs);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_open, Exports::open);                               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_close, Exports::close);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_contains, Exports::contains);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_touch, Exports::touch);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_assign, Exports::assign);                           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_release, Exports::release);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_begin, Exports::fill_begin);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_wait, Exports::fill_wait);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_landed, Exports::fill_landed);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_fill_end, Exports::fill_end);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_close_admission, Exports::close_admission);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_prefill_share, Exports::set_prefill_share);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_enable_copy_engine, Exports::enable_copy_engine);   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_copy_table, Exports::set_copy_table);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_arm_copy_engine, Exports::arm_copy_engine);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_enable_cpu_experts, Exports::enable_cpu_experts);   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_cpu_layer, Exports::set_cpu_layer);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_cpu_split, Exports::set_cpu_split);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_reserve_staging, Exports::reserve_staging);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_bulk_delta_count, Exports::bulk_delta_count);       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_take_bulk_delta, Exports::take_bulk_delta);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_cpu_stats, Exports::cpu_stats);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_copy_expert_bytes, Exports::copy_expert_bytes);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_calibrate_cpu_split, Exports::calibrate_cpu_split); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_mapping, Exports::mapping);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_slot_to_expert, Exports::slot_to_expert);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_lru_order, Exports::lru_order);                     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_set_hot, Exports::set_hot);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_counters, Exports::counters);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_group_counters, Exports::group_counters);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_core_counter_mask, Exports::core_counter_mask);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_layer_rows, Exports::layer_rows);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_words, Exports::trace_words);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_enable, Exports::trace_enable);               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_drain, Exports::trace_drain);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_trace_dropped, Exports::trace_dropped);             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_start_thread, Exports::start_thread);               \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_stop_thread, Exports::stop_thread);                 \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pause, Exports::pause);                             \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_resume, Exports::resume);                           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_cpu_open, Exports::draft_cpu_open);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_cpu_set_layer, Exports::draft_cpu_set_layer); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_cpu_start, Exports::draft_cpu_start);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_cpu_stop, Exports::draft_cpu_stop);           \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_draft_cpu_stats, Exports::draft_cpu_stats);
