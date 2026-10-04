// The full-stack CPU-expert benchmark: times the CPU-expert path above the kernel.
//
// A writer thread (the GPU's stand-in, DeviceSim) posts CPU-hit requests into the lease lanes. The real RamTier,
// RamThread and CpuExpertEngine serve them with the optimized EXL3 kernel, and the writer times post -> CopyDone
// against the bare kernel call on the same layer, slots, x and output memory. The difference is the stack's overhead.
//
//   Options / resolve_placement   the command line and where each thread runs
//   Bench                         the two phases: the bare forwards first, then the stack (see its doc)
//   bm_bare / bm_stack            the Google Benchmark bodies and their counters
//   main                          setup, --validate-only and --self-test, benchmark registration
//
// See python/sglang/kernels/jit/csrc/moe/expert_stream/bench/README.txt, "Full-stack bench".
#include <benchmark/benchmark.h>

#include "cpu_experts_cabi.h"
#include "device_sim.h"
#include "placement.h"
#include "self_test.h"
#include "stack.h"
#include "stack_fixture.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace es = ::sglang::expert_stream;
using namespace fullstack;
constexpr int kGroups = es::Wire::kNodes;

// The command line. Defaults match the reference machine's partition (README "Placement"); --self-test has its own.
struct Options {
  fs::path fixture = "/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin";
  fs::path references = "/data/models/exl3_exp/threading";
  fs::path image_dir = "/data/models/exl3_exp/google_benchmark/full-stack-images";
  std::optional<int> writer_cpu, copy_cpu;
  std::optional<std::string> service_cpus, cpus, worker_nodes;  // one entry per NUMA group, see resolve_placement
  int host_node = 0;
  int warmup = 128;
  int gap_us = 0;
  int keep_warm_us = 0;
  int wait_timeout_ms = 2000;
  bool validate_only = false;
  bool self_test = false;
};

// Consumes this bench's flags and leaves the rest (Google Benchmark's) in argv. Throws on an invalid value.
Options parse_options(int& argc, char** argv) {
  Options opt;
  int remaining = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
    if (arg.starts_with("--fixture="))
      opt.fixture = value("--fixture=");
    else if (arg.starts_with("--reference-dir="))
      opt.references = value("--reference-dir=");
    else if (arg.starts_with("--image-dir="))
      opt.image_dir = value("--image-dir=");
    else if (arg.starts_with("--writer-cpu="))
      opt.writer_cpu = number(value("--writer-cpu="));
    else if (arg.starts_with("--service-cpu="))
      opt.service_cpus = value("--service-cpu=");
    else if (arg.starts_with("--copy-cpu="))
      opt.copy_cpu = number(value("--copy-cpu="));
    else if (arg.starts_with("--cpus="))
      opt.cpus = value("--cpus=");
    else if (arg.starts_with("--host-node="))
      opt.host_node = number(value("--host-node="));
    else if (arg.starts_with("--worker-node="))
      opt.worker_nodes = value("--worker-node=");
    else if (arg.starts_with("--warmup-forwards="))
      opt.warmup = number(value("--warmup-forwards="));
    else if (arg.starts_with("--gap-us="))
      opt.gap_us = number(value("--gap-us="));
    else if (arg.starts_with("--keep-warm-us="))
      opt.keep_warm_us = number(value("--keep-warm-us="));
    else if (arg.starts_with("--wait-timeout-ms="))
      opt.wait_timeout_ms = number(value("--wait-timeout-ms="));
    else if (arg == "--validate-only")
      opt.validate_only = true;
    else if (arg == "--self-test")
      opt.self_test = true;
    else if (arg == "--help") {
      std::cout
          << "Full-stack CPU-expert benchmark (" EXL3_BENCH_BACKEND
             ")\n"
             "--fixture=FILE --reference-dir=DIR --image-dir=DIR (O_DIRECT-capable; row images are written there)\n"
             "--writer-cpu=16 --service-cpu=17 --copy-cpu=52 --cpus=18-33 --host-node=0 --worker-node=1\n"
             "one NUMA group per -DEXPERT_STREAM_NODES: --service-cpu and --worker-node take one entry per group, "
             "comma-separated, and --cpus one CPU list per group, '/'-separated: --service-cpu=17,35 "
             "--cpus=8-15/18-33 --worker-node=0,1\n"
             "--warmup-forwards=128 --gap-us=0 --keep-warm-us=0 --wait-timeout-ms=2000 --validate-only\n"
             "--keep-warm-us: the CPU expert thread's keep-warm window after each job (0: off)\n"
             "--self-test: synthetic rows and a fake forward; defaults --writer-cpu=0 --service-cpu=1 --copy-cpu=2 "
             "--cpus=3 (two groups: --copy-cpu=1 --service-cpu=2,3 --cpus=4/5)\n"
             "Google Benchmark flags are also accepted.\n";
      argv[remaining++] = argv[i];
    } else
      argv[remaining++] = argv[i];
  }
  argc = remaining;
  argv[remaining] = nullptr;
  if (opt.wait_timeout_ms < 2) throw std::runtime_error("--wait-timeout-ms must be at least 2");
  return opt;
}

// Splits `text` on `separator`; throws unless there are exactly kGroups entries, one per NUMA group, none empty.
std::vector<std::string> per_group(const std::string& text, char separator, const char* flag) {
  std::vector<std::string> entries;
  size_t begin = 0;
  for (;;) {  // not getline: it drops a trailing empty entry, so "18-33/" would pass at one group
    const size_t end = text.find(separator, begin);
    entries.push_back(text.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (static_cast<int>(entries.size()) != kGroups)
    throw std::runtime_error(
        std::string(flag) + ": give one entry per NUMA group (" + std::to_string(kGroups) + "), got " + text);
  for (const std::string& entry : entries)
    if (entry.empty()) throw std::runtime_error(std::string(flag) + ": an empty entry in " + text);
  return entries;
}

// Resolves the CPUs: at one group the defaults are production's placement; the self-test's fit any few CPUs (run it
// under taskset -c 0-15). Above one group the self-test has its own defaults and a measured run needs every per-group
// flag, since there is no reference placement to default to.
Placement resolve_placement(const Options& o) {
  Placement p;
  p.host_node = o.host_node;
  std::string services, cpus, nodes;
  if (kGroups == 1) {
    services = o.self_test ? "1" : "17";
    cpus = o.self_test ? "3" : "18-33";
    nodes = "1";
    p.writer = o.self_test ? 0 : 16;
    p.copy = o.self_test ? 2 : 52;
  } else if (o.self_test) {
    services = "2,3";
    cpus = "4/5";
    nodes = "0,1";
    p.writer = 0;
    p.copy = 1;
  } else {
    if (!o.service_cpus || !o.cpus || !o.worker_nodes)
      throw std::runtime_error(
          "a " + std::to_string(kGroups) + "-group run needs --service-cpu, --cpus and --worker-node (one entry per "
          "NUMA group)");
    p.writer = 16;
    p.copy = 52;
  }
  p.writer = o.writer_cpu.value_or(p.writer);
  p.copy = o.copy_cpu.value_or(p.copy);
  const auto service_list = per_group(o.service_cpus.value_or(services), ',', "--service-cpu");
  const auto cpu_lists = per_group(o.cpus.value_or(cpus), '/', "--cpus");
  const auto node_list = per_group(o.worker_nodes.value_or(nodes), ',', "--worker-node");
  for (int g = 0; g < kGroups; ++g)
    p.groups.push_back({number(service_list[g]), parse_cpus(cpu_lists[g]), number(node_list[g])});
  return p;
}

// The Stack's configuration for this fixture and placement: every eligible lane goes to the CPU, each group runs on its
// own engine and slot range, and the instrumented build gets a stage trace ring.
StackConfig stack_config(
    const StackFixture& f, const Placement& p, const Options& o, const std::vector<int64_t>& engines) {
  StackConfig c;
  c.rows = f.row_set();
  c.staging = StackFixture::kStaging;
  c.forward = &sglang_exl3_cpu_experts_forward;
  c.keep_warm = &sglang_exl3_cpu_experts_keep_warm;
  c.keep_warm_ns = static_cast<int64_t>(o.keep_warm_us) * 1000;
  c.x_base = f.x_row(0);
  c.x_stride = f.x_stride();
  c.out_base = reinterpret_cast<uint8_t*>(f.out_row(0));
  c.out_stride = f.out_stride();
  c.hidden = f.hidden();
  c.copy_cpu = p.copy;
  c.wait_timeout_ns = int64_t{o.wait_timeout_ms} * 1'000'000;
  for (int g = 0; g < kGroups; ++g) {
    StackConfig::Group group;
    group.service_cpu = p.groups[g].service;
    group.cores.assign(p.groups[g].workers.begin(), p.groups[g].workers.end());
    group.engine = engines[g];
    for (int n = 0; n <= es::Wire::kLanes; ++n)
      group.split[n] = n;
    c.groups.push_back(group);
    c.ranges.emplace_back(g * StackFixture::kGroupSlots, (g + 1) * StackFixture::kGroupSlots);
  }
  if constexpr (BenchBuild::kMetrics) c.trace_capacity = 4096;
  return c;
}

// Owns the registered CPU layer handles. Freed after the stack, whose CPU expert thread uses them, so declare it
// before the Bench that builds the Stack.
struct LayerHandles {
  std::vector<int64_t> handles;
  ~LayerHandles() {
    for (int64_t handle : handles)
      StackFixture::free_layer(handle);
  }
};

// One timed request: t0 is taken before the x store, t1 when CopyDone is seen.
struct StackCall {
  int64_t t0 = 0;
  int64_t t1 = 0;
  SimRequest request;
};

// The benchmark state, in two phases that must run in this order.
//
// Bare phase: the slots hold expert e in slot_of(e) (StackFixture::preload_slots) and no stack exists, so no CPU expert
// thread spins on worker 0's core, where the kernel puts every caller, and the kernel's per-caller OpenMP team is the
// only one. Stack phase (enter_stack_phase): the bare caller's team is released, the Stack is built, every expert is
// loaded through the tier's reader, and the layers are registered. A bare forward after that fails: it would share
// worker 0's core with the CPU expert thread and run a second OpenMP team.
//
// Runs on the writer's thread. Every path is checked bit-exactly against the frozen references (validate), and the
// thread census (check_threads) runs once per phase, before that phase is timed.
class Bench {
 public:
  Bench(
      const Options& options,
      Placement placement,
      StackFixture& fixture,
      std::vector<int64_t> handles,
      std::vector<int64_t> engines,
      std::set<int> before)
      : options_(options),
        placement_(std::move(placement)),
        fixture_(fixture),
        handles_(std::move(handles)),
        engines_(std::move(engines)),
        before_(std::move(before)),
        deadline_ns_(int64_t{options.wait_timeout_ms} * 1'000'000 / 2) {
    for (int k : {1, 3, 5}) {
      for (int i = 0; i < k; ++i) {
        experts_[k].push_back(i);
        // cpu_forward.cpp's routing coefficients, which the frozen references were computed with
        weights_[k].push_back(0.071234f + (k == 1 ? 0.0f : 0.23f * i / (k - 1)));
      }
    }
  }

  const Options& options() const {
    return options_;
  }
  const Placement& placement() const {
    return placement_;
  }
  int64_t rows() const {
    return fixture_.rows();
  }

  // Whether any of the k experts of a call is homed on `group`: only those groups run a CPU job.
  bool group_has_lanes(int k, int group) const {
    for (int32_t e : experts_.at(k))
      if (es::Wire::home(e) == group) return true;
    return false;
  }
  int groups_with_lanes(int k) const {
    int n = 0;
    for (int g = 0; g < kGroups; ++g)
      n += group_has_lanes(k, g) ? 1 : 0;
    return n;
  }

  Stack<BenchBuild>& stack() {
    enter_stack_phase();
    return *stack_;
  }

  // Checks thread affinity once per phase, before the phase is timed: a failed check must stop the run before any
  // timing. The bare phase has only the kernel's helpers (this thread is worker 0 and predates `before`); the stack
  // phase has every thread setup created. Call after a forward of the phase, which builds its OpenMP team.
  void check_threads() {
    if (stack_) {
      if (stack_census_) return;
      verify_threads(before_, expected_threads(placement_));
      stack_census_ = true;
    } else {
      if (bare_census_) return;
      // Above one group the bare forwards of every group share this thread's one OpenMP team, which each forward
      // re-pins to its own engine's cores: only the last group's helpers are left to count.
      const auto& workers = placement_.groups.back().workers;
      verify_threads(before_, std::vector<int>(workers.begin() + 1, workers.end()));
      bare_census_ = true;
    }
  }

  // Ends the bare phase: builds the stack and loads every expert into the slots the bare forwards used. Throws if the
  // tier picks different slots. Idempotent.
  void enter_stack_phase() {
    if (stack_) return;
    release_kernel_team();
    stack_ = std::make_unique<Stack<BenchBuild>>(stack_config(fixture_, placement_, options_, engines_));
    sim_ = std::make_unique<DeviceSim>(stack_->page(), stack_->lease(), rows(), fixture_.experts());
    const int64_t timeout_ns = int64_t{options_.wait_timeout_ms} * 1'000'000;
    // One load per group, so no post mixes groups' misses: each group's experts then arrive ascending in posts of its
    // kStaging staging slots, the order in which the tier maps a group's j-th expert at its j-th slot
    // (StackFixture::slot_of).
    std::vector<std::vector<int32_t>> by_group(kGroups);
    for (int32_t e = 0; e < fixture_.experts(); ++e)
      by_group[es::Wire::home(e)].push_back(e);
    for (int64_t row = 0; row < rows(); ++row) {
      for (const std::vector<int32_t>& group : by_group)
        if (!group.empty()) load_experts(*sim_, row, group, static_cast<int>(StackFixture::kStaging), timeout_ns);
      for (int32_t e = 0; e < fixture_.experts(); ++e)
        if (sim_->ram_slot(row, e) != StackFixture::slot_of(e))
          throw std::runtime_error(
              "row " + std::to_string(row) + ": the tier put expert " + std::to_string(e) + " in slot " +
              std::to_string(sim_->ram_slot(row, e)) + ", not the slot the bare forwards used");
      stack_->set_cpu_layer(row, handles_[row]);
      sim_->set_row_cpu(row);
    }
  }

  // The device's part of one request: x store, record, copy wait. [t0, t1] is the timed interval. Throws if the wait
  // passes its deadline or a lane is not typed HIT_CPU.
  StackCall stack_call(int64_t row, int k) {
    enter_stack_phase();
    StackCall call;
    call.t0 = monotonic_ns();
    fixture_.write_x(row);
    call.request = sim_->post(row, experts_[k], weights_[k], /*captured=*/true, call.t0 + deadline_ns_);
    const bool done = sim_->copy_wait(call.request, call.t0 + deadline_ns_);
    call.t1 = monotonic_ns();
    if (!done) throw std::runtime_error("the copy wait passed its deadline: " + describe(call.request));
    for (int j = 0; j < k; ++j) {
      if (call.request.kinds[j] != static_cast<int32_t>(es::Wire::kKindHitCpu))
        throw std::runtime_error("lane " + std::to_string(j) + " was not typed HIT_CPU: " + describe(call.request));
    }
    return call;
  }

  // The C ABI forward on the same layer handle, slots and x as the stack, from the calling thread, over the experts of
  // experts_[k] homed on `group` (their slots are slot_of(expert)), on that group's engine and worker count, into
  // `out`. Does nothing when the group has none. Throws once the stack exists (see the class doc).
  void bare_into(int64_t row, int k, int group, float* out) {
    if (stack_)
      throw std::runtime_error(
          "a bare forward once the stack exists would share worker 0's core with the CPU expert "
          "thread and run a second OpenMP team: BM_bare runs first "
          "(no --benchmark_enable_random_interleaving)");
    std::vector<int32_t> slots;
    std::vector<float> weights;
    for (int i = 0; i < k; ++i) {
      if (es::Wire::home(experts_[k][i]) != group) continue;
      slots.push_back(StackFixture::slot_of(experts_[k][i]));
      weights.push_back(weights_[k][i]);
    }
    if (slots.empty()) return;
    SglangCpuExpertsForward call{};
    call.abi_version = SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION;
    call.rows = 1;
    call.layer = handles_[row];
    call.x = fixture_.x_row(row);
    call.slots = slots.data();
    call.weights = weights.data();
    call.out = out;
    call.k = static_cast<int32_t>(slots.size());
    call.threads = static_cast<int32_t>(placement_.groups[group].workers.size());
    call.engine = engines_[group];
    if (sglang_exl3_cpu_experts_forward(&call) != 0)
      throw std::runtime_error("the bare CPU forward failed");
  }

  // bare_into the group's part 0 of the row's output, where the stack's CPU expert thread writes it.
  void bare_call(int64_t row, int k, int group = 0) {
    bare_into(row, k, group, fixture_.out_row(row) + 2 * group * fixture_.hidden());
  }

  // Every layer's output for k experts, through the stack or bare, against reference-e{k}.bin, bit-exact. Each output
  // is NaN-poisoned first, so a forward that did not run fails. Above one group only the stack path exists, and each
  // group's part is checked as validate_groups says.
  void validate(int k, bool via_stack) {
    if (kGroups > 1) {
      if (!via_stack) throw std::runtime_error("above one NUMA group the bare forwards are record_bare's");
      validate_groups(k);
      return;
    }
    const int64_t hidden = fixture_.hidden();
    std::vector<float> results(static_cast<size_t>(rows() * hidden));
    for (int64_t row = 0; row < rows(); ++row) {
      float* out = fixture_.out_row(row);
      std::fill(out, out + hidden, std::numeric_limits<float>::quiet_NaN());
      if (via_stack) {
        stack_call(row, k);
      } else {
        bare_call(row, k);
      }
      for (int64_t h = 0; h < hidden; ++h)
        if (!std::isfinite(out[h])) throw std::runtime_error("non-finite output in layer " + std::to_string(row));
      std::copy(out, out + hidden, results.begin() + row * hidden);
    }
    check_reference(options_.references / ("reference-e" + std::to_string(k) + ".bin"), results);
  }

  // Above one group: every row's bare forward of each group's experts, on that group's engine and worker 0, kept for
  // validate_groups. Before the stack exists.
  void record_bare(int k) {
    const int64_t hidden = fixture_.hidden();
    std::vector<float>& bare = bare_[k];
    bare.assign(static_cast<size_t>(rows() * es::Wire::kNodes * hidden), 0.0f);
    for (int g = 0; g < es::Wire::kNodes; ++g) {
      if (!group_has_lanes(k, g)) continue;
      PinScope caller(placement_.groups[g].workers.front());
      for (int64_t row = 0; row < rows(); ++row)
        bare_into(row, k, g, bare.data() + (row * es::Wire::kNodes + g) * hidden);
    }
  }

  // Above one group: each group's part of the stack's output equals, bit for bit, its bare forward; the parts' sum
  // matches the single-engine reference to within fp32 reassociation.
  void validate_groups(int k) {
    const int64_t hidden = fixture_.hidden();
    const std::vector<float>& bare = bare_.at(k);
    std::vector<float> stacked(static_cast<size_t>(rows() * hidden), 0.0f);
    for (int64_t row = 0; row < rows(); ++row) {
      float* out = fixture_.out_row(row);
      std::fill(out, out + 2 * es::Wire::kNodes * hidden, std::numeric_limits<float>::quiet_NaN());
      stack_call(row, k);
      for (int g = 0; g < es::Wire::kNodes; ++g) {
        if (!group_has_lanes(k, g)) continue;
        const float* part = out + 2 * g * hidden;
        if (std::memcmp(part, bare.data() + (row * es::Wire::kNodes + g) * hidden, hidden * sizeof(float)) != 0)
          throw std::runtime_error(
              "row " + std::to_string(row) + ": group " + std::to_string(g) + "'s stack part differs from its bare forward");
        for (int64_t h = 0; h < hidden; ++h)
          stacked[row * hidden + h] += part[h];
      }
    }
    check_reference_close(options_.references / ("reference-e" + std::to_string(k) + ".bin"), stacked, 1e-5f);
  }

  // A one-line account of a request and the stack's state, for error messages.
  std::string describe(const SimRequest& r) {
    std::ostringstream s;
    s << "gen " << r.gen << " (seq " << r.seq << ") row " << r.row << ", " << r.count << " lanes, kinds";
    for (int j = 0; j < r.count; ++j)
      s << ' ' << r.kinds[j];
    const auto c = stack_->counters();
    s << "; served " << c[es::kServedRequests] << " touch_only " << c[es::kTouchOnly] << " rows_read "
      << c[es::kRowsRead] << " overruns " << c[es::kOverruns] << "; cpu jobs/lanes";
    for (int g = 0; g < kGroups; ++g) {
      const auto cpu = stack_->cpu_stats(g);
      s << " g" << g << ' ' << cpu[0] << '/' << cpu[1];
    }
    s << "; CopyDone " << sim_->copy_done(r) << ", gate 0x" << std::hex << sim_->copy_gate() << std::dec
      << ", handled through " << stack_->tier().handled_through();
    return s.str();
  }

  std::map<int, double> bare_p50_us;  // BM_bare's p50 per k, for BM_stack's overhead counter

 private:
  const Options& options_;
  Placement placement_;
  StackFixture& fixture_;
  std::vector<int64_t> handles_;  // owned by main's LayerHandles, which outlives this
  std::vector<int64_t> engines_;  // one kernel engine per group, on that group's workers
  std::set<int> before_;          // the process's threads before setup
  bool bare_census_ = false;
  bool stack_census_ = false;
  int64_t deadline_ns_;
  std::map<int, std::vector<int32_t>> experts_;
  std::map<int, std::vector<float>> weights_;
  std::map<int, std::vector<float>> bare_;  // record_bare's [row][group][hidden] per k
  std::unique_ptr<Stack<BenchBuild>> stack_;  // built by enter_stack_phase; null during the bare phase
  std::unique_ptr<DeviceSim> sim_;
};

bool benchmark_failed = false;

// The `fraction` quantile of `seconds`, in microseconds (nearest rank, no interpolation).
double quantile_us(std::vector<double> seconds, double fraction) {
  const size_t index = static_cast<size_t>(fraction * static_cast<double>(seconds.size() - 1));
  std::nth_element(seconds.begin(), seconds.begin() + static_cast<std::ptrdiff_t>(index), seconds.end());
  return seconds[index] * 1e6;
}

// Sleeps before a timed call when --gap-us is set, to test wakeup behaviour; outside the timed interval.
void gap(const Options& options) {
  if (options.gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us));
}

// Once one benchmark has failed, the rest report nothing: a failed process yields no timings.
bool skip_after_failure(benchmark::State& state) {
  if (!benchmark_failed) return false;
  state.SkipWithError("skipped: an earlier benchmark in this process failed");
  return true;
}

// BM_bare/experts:k: the C ABI forward called from worker 0's CPU. Counters: p50/p95/p99 per call.
void bm_bare(benchmark::State& state, Bench& bench, int k) {
  if (skip_after_failure(state)) return;
  try {
    PinScope caller(bench.placement().groups[0].workers.front());  // the kernel runs its caller as worker 0, on that core
    bench.validate(k, false);
    bench.check_threads();
    for (int i = 0; i < bench.options().warmup; ++i)
      bench.bare_call(i % bench.rows(), k);
    std::vector<double> samples;
    int64_t row = 0;
    for (auto _ : state) {
      gap(bench.options());
      const int64_t t0 = monotonic_ns();
      bench.bare_call(row, k);
      const int64_t t1 = monotonic_ns();
      const double seconds = static_cast<double>(t1 - t0) * 1e-9;
      state.SetIterationTime(seconds);
      samples.push_back(seconds);  // sample storage and layer selection are outside the timed interval
      row = (row + 1) % bench.rows();
    }
    bench.validate(k, false);
    if (!samples.empty()) {
      const double p50 = quantile_us(samples, 0.50);
      state.counters["p50_us"] = p50;
      state.counters["p95_us"] = quantile_us(samples, 0.95);
      state.counters["p99_us"] = quantile_us(samples, 0.99);
      bench.bare_p50_us[k] = p50;
    }
    state.counters["experts"] = k;
    state.counters["workers"] = static_cast<double>(bench.placement().groups[0].workers.size());
    state.counters["layers"] = static_cast<double>(bench.rows());
  } catch (const std::exception& error) {
    benchmark_failed = true;
    state.SkipWithError(error.what());
  }
}

// Every group's CPU expert engine totals, {jobs, lanes, forward ns} each.
using CpuTotals = std::array<std::array<int64_t, 3>, kGroups>;
CpuTotals cpu_totals(const Stack<BenchBuild>& stack) {
  CpuTotals totals;
  for (int g = 0; g < kGroups; ++g)
    totals[g] = stack.cpu_stats(g);
  return totals;
}

// BM_stack/experts:k: x store, record, gate close, spin until CopyDone. Counters: p50/p95/p99 per call and
// overhead_p50_us against BM_bare; the instrumented build adds the stage breakdown (see README "Full-stack bench").
void bm_stack(benchmark::State& state, Bench& bench, int k) {
  if (skip_after_failure(state)) return;
  try {
    Stack<BenchBuild>& stack = bench.stack();
    bench.validate(k, true);
    bench.check_threads();
    for (int i = 0; i < bench.options().warmup; ++i)
      bench.stack_call(i % bench.rows(), k);
    std::vector<double> total, pickup, service, forward, handoff;
    [[maybe_unused]] auto stage = std::make_unique<es::StageRecord>();
    if constexpr (BenchBuild::kMetrics) stack.drain_all();
    const auto counters0 = stack.counters();
    const auto cpu0 = cpu_totals(stack);
    int64_t calls = 0;
    int64_t row = 0;
    for (auto _ : state) {
      gap(bench.options());
      const auto cpu_before = cpu_totals(stack);
      const StackCall call = bench.stack_call(row, k);
      const double seconds = static_cast<double>(call.t1 - call.t0) * 1e-9;
      state.SetIterationTime(seconds);
      total.push_back(seconds);
      if constexpr (BenchBuild::kMetrics) {
        // The breakdown, from the stage trace and the CPU expert thread's forward time (exact: one request in flight).
        if (!stack.drain_stage(*stage, call.request.seq, call.t1 + 1'000'000'000))
          throw std::runtime_error("no stage record for request " + std::to_string(call.request.seq));
        // The groups' forwards run at once: the request waits for the slowest.
        const auto cpu_after = cpu_totals(stack);
        int64_t forward_ns = 0;
        for (int g = 0; g < kGroups; ++g)
          forward_ns = std::max(forward_ns, cpu_after[g][2] - cpu_before[g][2]);
        const double forward_s = static_cast<double>(forward_ns) * 1e-9;
        pickup.push_back(static_cast<double>(stage->observed - call.t0) * 1e-9);
        service.push_back(static_cast<double>(stage->done - stage->observed) * 1e-9);
        forward.push_back(forward_s);
        handoff.push_back(static_cast<double>(call.t1 - stage->done) * 1e-9 - forward_s);
      }
      ++calls;
      row = (row + 1) % bench.rows();
    }
    // The timed calls must be exactly the CPU jobs: one job per group holding lanes per call, k lanes in all, no row
    // read, no overrun.
    const auto counters1 = stack.counters();
    const auto cpu1 = cpu_totals(stack);
    int64_t jobs = 0, lanes = 0;
    for (int g = 0; g < kGroups; ++g) {
      jobs += cpu1[g][0] - cpu0[g][0];
      lanes += cpu1[g][1] - cpu0[g][1];
    }
    if (jobs != calls * bench.groups_with_lanes(k) || lanes != calls * k)
      throw std::runtime_error(
          "CPU jobs/lanes " + std::to_string(jobs) + "/" + std::to_string(lanes) + " for " + std::to_string(calls) +
          " calls of " + std::to_string(k) + " lanes over " + std::to_string(bench.groups_with_lanes(k)) + " groups");
    if (counters1[es::kRowsRead] != counters0[es::kRowsRead] || counters1[es::kOverruns] != counters0[es::kOverruns])
      throw std::runtime_error("a row read or an overrun during timing");
    if constexpr (BenchBuild::kMetrics) {
      if (stack.tier().trace_dropped() != 0) throw std::runtime_error("the stage trace dropped records");
    }
    bench.validate(k, true);
    if (!total.empty()) {
      const double p50 = quantile_us(total, 0.50);
      state.counters["p50_us"] = p50;
      state.counters["p95_us"] = quantile_us(total, 0.95);
      state.counters["p99_us"] = quantile_us(total, 0.99);
      if (bench.bare_p50_us.contains(k)) state.counters["overhead_p50_us"] = p50 - bench.bare_p50_us[k];
      if constexpr (BenchBuild::kMetrics) {
        state.counters["pickup_p50_us"] = quantile_us(pickup, 0.50);
        state.counters["pickup_p95_us"] = quantile_us(pickup, 0.95);
        state.counters["service_p50_us"] = quantile_us(service, 0.50);
        state.counters["service_p95_us"] = quantile_us(service, 0.95);
        const double forward_p50 = quantile_us(forward, 0.50);
        state.counters["forward_p50_us"] = forward_p50;
        state.counters["forward_p95_us"] = quantile_us(forward, 0.95);
        // The bare baseline runs before the stack exists, with no service, writer or copy thread spinning: these
        // show how far the in-stack kernel time moved from it, and the overhead against the in-stack kernel itself.
        state.counters["overhead_vs_forward_p50_us"] = p50 - forward_p50;
        if (bench.bare_p50_us.contains(k))
          state.counters["forward_vs_bare_p50_us"] = forward_p50 - bench.bare_p50_us[k];
        state.counters["handoff_p50_us"] = quantile_us(handoff, 0.50);
        state.counters["handoff_p95_us"] = quantile_us(handoff, 0.95);
      }
    }
    state.counters["experts"] = k;
    size_t workers = 0;
    for (const GroupPlacement& group : bench.placement().groups)
      workers += group.workers.size();
    state.counters["workers"] = static_cast<double>(workers);
    state.counters["groups"] = kGroups;
    state.counters["layers"] = static_cast<double>(bench.rows());
  } catch (const std::exception& error) {
    benchmark_failed = true;
    state.SkipWithError(error.what());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    const Placement placement = resolve_placement(options);
    validate_placement(placement, system_topology(), !options.self_test);
    if (options.self_test) return run_self_test(placement, options.image_dir) == 0 ? 0 : 1;

    setenv("EXL3_MOE_CPU_PIN", "0", 1);
    setenv("EXL3_MOE_CPU_SMALL_WORKERS", "0", 1);
    configure_cpu_kernel_runtime();
    std::vector<int64_t> engines(kGroups, 0);  // the kernel's engine on each group's workers: the stack and the bare forwards run on it
    std::vector<int> group_nodes;
    for (int g = 0; g < kGroups; ++g) {
      const auto& workers = placement.groups[g].workers;
      if (sglang_exl3_cpu_experts_engine_create(workers.data(), static_cast<int32_t>(workers.size()), &engines[g]) != 0)
        throw std::runtime_error("Cannot create the kernel's engine for group " + std::to_string(g));
      group_nodes.push_back(placement.groups[g].node);
    }
    const auto before = task_ids();
    std::unique_ptr<StackFixture> fixture;
    {
      PinScope first_touch(placement.groups[0].workers.front());  // x and outputs on group 0's node; each group's slabs are bound to its own
      fixture = std::make_unique<StackFixture>(options.fixture, options.image_dir, group_nodes);
    }
    pin_self(placement.writer);  // the writer's stores and clock; the stack's page and lease are first-touched here
    LayerHandles layers;         // outlives the Bench, whose stack's CPU expert thread uses them
    for (int64_t row = 0; row < fixture->rows(); ++row) {
      fixture->preload_slots(row);
      layers.handles.push_back(fixture->register_layer(row));
    }
    Bench bench(options, placement, *fixture, layers.handles, engines, before);
    const std::vector<int> expected = expected_threads(placement);
    if (kGroups > 1) {
      // Every group's bare forwards before the stack exists: a bare forward after it would share worker 0's core with
      // a CPU expert thread.
      for (int k : {1, 3, 5})
        bench.record_bare(k);
      bench.check_threads();
    }
    if (options.validate_only) {
      if (kGroups == 1) {
        PinScope caller(placement.groups[0].workers.front());  // the kernel runs its caller as worker 0, on that core
        for (int k : {1, 3, 5})
          bench.validate(k, /*via_stack=*/false);
        bench.check_threads();
      }
      for (int k : {1, 3, 5})
        bench.validate(k, /*via_stack=*/true);  // the first builds the stack
      bench.check_threads();
      if (kGroups == 1)
        std::cerr << "Verified 48 bit-exact layer outputs (24 through the stack, 24 bare); threads pinned to {"
                  << cpu_list(expected) << "}; writer CPU " << placement.writer << " (" << BenchBuild::kName
                  << " build)\n";
      else
        std::cerr << "Verified 24 layer outputs through the stack on " << kGroups
                  << " NUMA groups (each group's part bit-exact against its bare forward, the parts' sum against the "
                     "reference within 1e-5); threads pinned to {"
                  << cpu_list(expected) << "}; writer CPU " << placement.writer << " (" << BenchBuild::kName
                  << " build)\n";
      return 0;
    }
    benchmark::AddCustomContext("backend", EXL3_BENCH_BACKEND);
    benchmark::AddCustomContext("host_build", std::string(BenchBuild::kName));
    benchmark::AddCustomContext("fixture", options.fixture.string());
    benchmark::AddCustomContext("image_dir", options.image_dir.string());
    benchmark::AddCustomContext("writer_cpu", std::to_string(placement.writer));
    benchmark::AddCustomContext("copy_cpu", std::to_string(placement.copy));
    for (int g = 0; g < kGroups; ++g) {
      const std::string suffix = kGroups > 1 ? "_g" + std::to_string(g) : "";
      const GroupPlacement& group = placement.groups[g];
      benchmark::AddCustomContext("service_cpu" + suffix, std::to_string(group.service));
      benchmark::AddCustomContext(
          "worker_cpus" + suffix, cpu_list(std::vector<int>(group.workers.begin(), group.workers.end())));
    }
    std::ifstream cgroup_file("/proc/self/cgroup");
    benchmark::AddCustomContext("cgroup", std::string((std::istreambuf_iterator<char>(cgroup_file)), {}));
    benchmark::AddCustomContext("gap_us", std::to_string(options.gap_us));
    benchmark::AddCustomContext("keep_warm_us", std::to_string(options.keep_warm_us));
    benchmark::AddCustomContext("compiler", __VERSION__);
    // Every BM_bare before any BM_stack: one OpenMP team at a time (see Bench). Each benchmark checks its path's 8
    // outputs bit-exactly before and after it is timed. Above one group there is no BM_bare: record_bare ran each
    // group's bare forwards above.
    if (kGroups == 1)
      for (int k : {1, 3, 5})
        benchmark::RegisterBenchmark(
            "BM_bare/experts:" + std::to_string(k), [&bench, k](benchmark::State& state) { bm_bare(state, bench, k); })
            ->UseManualTime()
            ->Unit(benchmark::kMicrosecond);
    for (int k : {1, 3, 5})
      benchmark::RegisterBenchmark(
          "BM_stack/experts:" + std::to_string(k), [&bench, k](benchmark::State& state) { bm_stack(state, bench, k); })
          ->UseManualTime()
          ->Unit(benchmark::kMicrosecond);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    bench.validate(1, /*via_stack=*/true);  // the census counts the stack's threads, even after a BM_bare-only filter
    verify_threads(before, expected);
    return benchmark_failed ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
