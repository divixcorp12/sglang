// The full-stack CPU-expert benchmark (docs/superpowers/specs/2026-10-01-expert-stream-full-stack-bench-design.md):
// a writer thread posts CPU-hit requests into the lease lanes; the real RamTier, RamThread and CpuExpertEngine serve
// them with the optimized EXL3 kernel; the writer times post -> CopyDone against the bare kernel call.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include "cpu_experts_cabi.h"
#include "device_sim.h"
#include "stack.h"
#include "stack_fixture.h"

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "placement.h"
#include "self_test.h"

namespace {
namespace fs = std::filesystem;
namespace es = ::sglang::expert_stream;
using namespace fullstack;

struct Options {
  fs::path fixture = "/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin";
  fs::path references = "/data/models/exl3_exp/threading";
  fs::path image_dir = "/data/models/exl3_exp/google_benchmark/full-stack-images";
  std::optional<int> writer_cpu, service_cpu, copy_cpu;
  std::optional<std::string> cpus;
  int host_node = 0;
  int worker_node = 1;
  int warmup = 128;
  int gap_us = 0;
  int wait_timeout_ms = 2000;
  bool validate_only = false;
  bool self_test = false;
};

Options parse_options(int& argc, char** argv) {
  Options opt;
  int remaining = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
    if (arg.starts_with("--fixture=")) opt.fixture = value("--fixture=");
    else if (arg.starts_with("--reference-dir=")) opt.references = value("--reference-dir=");
    else if (arg.starts_with("--image-dir=")) opt.image_dir = value("--image-dir=");
    else if (arg.starts_with("--writer-cpu=")) opt.writer_cpu = number(value("--writer-cpu="));
    else if (arg.starts_with("--service-cpu=")) opt.service_cpu = number(value("--service-cpu="));
    else if (arg.starts_with("--copy-cpu=")) opt.copy_cpu = number(value("--copy-cpu="));
    else if (arg.starts_with("--cpus=")) opt.cpus = value("--cpus=");
    else if (arg.starts_with("--host-node=")) opt.host_node = number(value("--host-node="));
    else if (arg.starts_with("--worker-node=")) opt.worker_node = number(value("--worker-node="));
    else if (arg.starts_with("--warmup-forwards=")) opt.warmup = number(value("--warmup-forwards="));
    else if (arg.starts_with("--gap-us=")) opt.gap_us = number(value("--gap-us="));
    else if (arg.starts_with("--wait-timeout-ms=")) opt.wait_timeout_ms = number(value("--wait-timeout-ms="));
    else if (arg == "--validate-only") opt.validate_only = true;
    else if (arg == "--self-test") opt.self_test = true;
    else if (arg == "--help") {
      std::cout << "Full-stack CPU-expert benchmark (" EXL3_BENCH_BACKEND ")\n"
        "--fixture=FILE --reference-dir=DIR --image-dir=DIR (O_DIRECT-capable; row images are written there)\n"
        "--writer-cpu=16 --service-cpu=17 --copy-cpu=52 --cpus=18-33 --host-node=0 --worker-node=1\n"
        "--warmup-forwards=128 --gap-us=0 --wait-timeout-ms=2000 --validate-only\n"
        "--self-test: synthetic rows and a fake forward; defaults --writer-cpu=0 --service-cpu=1 --copy-cpu=2 --cpus=3\n"
        "Google Benchmark flags are also accepted.\n";
      argv[remaining++] = argv[i];
    } else argv[remaining++] = argv[i];
  }
  argc = remaining;
  argv[remaining] = nullptr;
  if (opt.wait_timeout_ms < 2) throw std::runtime_error("--wait-timeout-ms must be at least 2");
  return opt;
}

// The bench's defaults are production's placement; the self-test's fit any four CPUs (run it under taskset -c 0-15).
Placement resolve_placement(const Options& o) {
  Placement p;
  p.writer = o.writer_cpu.value_or(o.self_test ? 0 : 16);
  p.service = o.service_cpu.value_or(o.self_test ? 1 : 17);
  p.copy = o.copy_cpu.value_or(o.self_test ? 2 : 52);
  p.workers = parse_cpus(o.cpus.value_or(o.self_test ? "3" : "18-33"));
  p.host_node = o.host_node;
  p.worker_node = o.worker_node;
  return p;
}

StackConfig stack_config(const StackFixture& f, const Placement& p, const Options& o) {
  StackConfig c;
  c.rows = f.row_set();
  c.staging = StackFixture::kStaging;
  c.forward = &sglang_exl3_cpu_experts_forward;
  c.threads = static_cast<int>(p.workers.size());
  c.cores.assign(p.workers.begin(), p.workers.end());
  c.x_base = f.x_row(0);
  c.x_stride = f.x_stride();
  c.out_base = reinterpret_cast<uint8_t*>(f.out_row(0));
  c.out_stride = f.out_stride();
  c.hidden = f.hidden();
  c.service_cpu = p.service;
  c.copy_cpu = p.copy;
  c.wait_timeout_ns = int64_t{o.wait_timeout_ms} * 1'000'000;
  for (int n = 0; n <= es::kLeaseLanes; ++n) c.split[n] = n;  // every eligible lane is the CPU's
  if constexpr (BenchBuild::kMetrics) c.trace_capacity = 4096;
  return c;
}

// Freed after the stack, whose CPU expert thread uses them: declare before the Stack.
struct LayerHandles {
  std::vector<int64_t> handles;
  ~LayerHandles() {
    for (int64_t handle : handles) StackFixture::free_layer(handle);
  }
};

struct StackCall {
  int64_t t0 = 0;  // before the x store
  int64_t t1 = 0;  // CopyDone seen
  SimRequest request;
};

class Bench {
 public:
  // The bare phase first: the slots hold expert e in slot e (StackFixture::preload_slots), and no stack exists, so no
  // CPU expert thread spins on worker 0's core, where the kernel puts every caller.
  Bench(const Options& options, Placement placement, StackFixture& fixture, std::vector<int64_t> handles,
        std::set<int> before)
      : options_(options),
        placement_(std::move(placement)),
        fixture_(fixture),
        handles_(std::move(handles)),
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

  const Options& options() const { return options_; }
  const Placement& placement() const { return placement_; }
  int64_t rows() const { return fixture_.rows(); }

  Stack<BenchBuild>& stack() {
    enter_stack_phase();
    return *stack_;
  }

  // The thread census, once per phase and before that phase is timed (the spec refuses a failed affinity check before
  // any timing). The bare phase has only the kernel's helpers (this thread is worker 0, and predates `before`); the
  // stack phase has every thread setup created. Call after a forward of the phase, which builds its OpenMP team.
  void check_threads() {
    if (stack_) {
      if (stack_census_) return;
      verify_threads(before_, expected_threads(placement_));
      stack_census_ = true;
    } else {
      if (bare_census_) return;
      verify_threads(before_, std::vector<int>(placement_.workers.begin() + 1, placement_.workers.end()));
      bare_census_ = true;
    }
  }

  // Ends the bare phase, on the writer's thread: release the bare caller's OpenMP team (one team at a time), build the
  // stack, load every expert through the tier's reader (into the slots the bare forwards used), register the layers.
  void enter_stack_phase() {
    if (stack_) return;
    release_kernel_team();
    stack_ = std::make_unique<Stack<BenchBuild>>(stack_config(fixture_, placement_, options_));
    sim_ = std::make_unique<DeviceSim>(stack_->page(), stack_->lease(), rows(), fixture_.experts());
    const int64_t timeout_ns = int64_t{options_.wait_timeout_ms} * 1'000'000;
    std::vector<int32_t> all(static_cast<size_t>(fixture_.experts()));
    std::iota(all.begin(), all.end(), 0);
    for (int64_t row = 0; row < rows(); ++row) {
      load_experts(*sim_, row, all, static_cast<int>(StackFixture::kStaging), timeout_ns);
      for (int32_t e : all)
        if (sim_->ram_slot(row, e) != e)
          throw std::runtime_error("row " + std::to_string(row) + ": the tier put expert " + std::to_string(e) +
                                   " in slot " + std::to_string(sim_->ram_slot(row, e)) +
                                   ", not the slot the bare forwards used");
      stack_->set_cpu_layer(row, handles_[row]);
      sim_->set_row_cpu(row);
    }
  }

  // The device's part of one request: x, the record, the copy wait. t0..t1 is the timed interval.
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
      if (call.request.kinds[j] != static_cast<int32_t>(es::kKindHitCpu))
        throw std::runtime_error("lane " + std::to_string(j) + " was not typed HIT_CPU: " + describe(call.request));
    }
    return call;
  }

  // The C ABI forward on the same layer handle, slots, x and output memory, from the calling thread. Slot e holds
  // expert e, so the experts are the slots.
  void bare_call(int64_t row, int k) {
    if (stack_)
      throw std::runtime_error("a bare forward once the stack exists would share worker 0's core with the CPU expert "
                               "thread and run a second OpenMP team: BM_bare runs first "
                               "(no --benchmark_enable_random_interleaving)");
    if (sglang_exl3_cpu_experts_forward(handles_[row], fixture_.x_row(row), experts_[k].data(), weights_[k].data(), k,
                                        fixture_.out_row(row), static_cast<int32_t>(placement_.workers.size()), 0) != 0)
      throw std::runtime_error("the bare CPU forward failed");
  }

  // Every layer's output for k experts, through the stack or bare, against reference-e{k}.bin, bit-exact. Each output
  // is NaN-poisoned first, so a forward that did not run fails.
  void validate(int k, bool via_stack) {
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

  std::string describe(const SimRequest& r) {
    std::ostringstream s;
    s << "gen " << r.gen << " (seq " << r.seq << ") row " << r.row << ", " << r.count << " lanes, kinds";
    for (int j = 0; j < r.count; ++j) s << ' ' << r.kinds[j];
    const auto c = stack_->counters();
    const auto cpu = stack_->cpu_stats();
    s << "; served " << c[es::kServedRequests] << " touch_only " << c[es::kTouchOnly] << " rows_read "
      << c[es::kRowsRead] << " overruns " << c[es::kOverruns] << "; cpu jobs " << cpu[0] << " lanes " << cpu[1]
      << "; CopyDone " << sim_->copy_done(r) << ", gate 0x" << std::hex << sim_->copy_gate() << std::dec
      << ", handled through " << stack_->tier().handled_through();
    return s.str();
  }

  std::map<int, double> bare_p50_us;  // BM_bare's p50 per k, for BM_stack's overhead counter

 private:
  const Options& options_;
  Placement placement_;
  StackFixture& fixture_;
  std::vector<int64_t> handles_;  // owned by main's LayerHandles, which outlives this
  std::set<int> before_;          // the process's threads before setup
  bool bare_census_ = false;
  bool stack_census_ = false;
  int64_t deadline_ns_;
  std::map<int, std::vector<int32_t>> experts_;
  std::map<int, std::vector<float>> weights_;
  std::unique_ptr<Stack<BenchBuild>> stack_;  // built by enter_stack_phase; null during the bare phase
  std::unique_ptr<DeviceSim> sim_;
};

bool benchmark_failed = false;

double quantile_us(std::vector<double> seconds, double fraction) {
  const size_t index = static_cast<size_t>(fraction * static_cast<double>(seconds.size() - 1));
  std::nth_element(seconds.begin(), seconds.begin() + static_cast<std::ptrdiff_t>(index), seconds.end());
  return seconds[index] * 1e6;
}

void gap(const Options& options) {
  if (options.gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(options.gap_us));
}

// No timing from a failed process: once one benchmark has failed, the rest report nothing.
bool skip_after_failure(benchmark::State& state) {
  if (!benchmark_failed) return false;
  state.SkipWithError("skipped: an earlier benchmark in this process failed");
  return true;
}

void bm_bare(benchmark::State& state, Bench& bench, int k) {
  if (skip_after_failure(state)) return;
  try {
    PinScope caller(bench.placement().workers.front());  // the kernel runs its caller as worker 0, on that core
    bench.validate(k, false);
    bench.check_threads();
    for (int i = 0; i < bench.options().warmup; ++i) bench.bare_call(i % bench.rows(), k);
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
    state.counters["workers"] = static_cast<double>(bench.placement().workers.size());
    state.counters["layers"] = static_cast<double>(bench.rows());
  } catch (const std::exception& error) {
    benchmark_failed = true;
    state.SkipWithError(error.what());
  }
}

void bm_stack(benchmark::State& state, Bench& bench, int k) {
  if (skip_after_failure(state)) return;
  try {
    Stack<BenchBuild>& stack = bench.stack();
    bench.validate(k, true);
    bench.check_threads();
    for (int i = 0; i < bench.options().warmup; ++i) bench.stack_call(i % bench.rows(), k);
    std::vector<double> total, pickup, service, forward, handoff;
    [[maybe_unused]] auto stage = std::make_unique<es::StageRecord>();
    if constexpr (BenchBuild::kMetrics) stack.drain_all();
    const auto counters0 = stack.counters();
    const auto cpu0 = stack.cpu_stats();
    int64_t calls = 0;
    int64_t row = 0;
    for (auto _ : state) {
      gap(bench.options());
      const auto cpu_before = stack.cpu_stats();
      const StackCall call = bench.stack_call(row, k);
      const double seconds = static_cast<double>(call.t1 - call.t0) * 1e-9;
      state.SetIterationTime(seconds);
      total.push_back(seconds);
      if constexpr (BenchBuild::kMetrics) {
        // The breakdown, from the stage trace and the CPU expert thread's forward time (exact: one request in flight).
        if (!stack.drain_stage(*stage, call.request.seq, call.t1 + 1'000'000'000))
          throw std::runtime_error("no stage record for request " + std::to_string(call.request.seq));
        const auto cpu_after = stack.cpu_stats();
        const double forward_s = static_cast<double>(cpu_after[2] - cpu_before[2]) * 1e-9;
        pickup.push_back(static_cast<double>(stage->observed - call.t0) * 1e-9);
        service.push_back(static_cast<double>(stage->done - stage->observed) * 1e-9);
        forward.push_back(forward_s);
        handoff.push_back(static_cast<double>(call.t1 - stage->done) * 1e-9 - forward_s);
      }
      ++calls;
      row = (row + 1) % bench.rows();
    }
    // Reconcile: one CPU job of k lanes per call, no read, no overrun.
    const auto counters1 = stack.counters();
    const auto cpu1 = stack.cpu_stats();
    if (cpu1[0] - cpu0[0] != calls || cpu1[1] - cpu0[1] != calls * k)
      throw std::runtime_error("CPU jobs/lanes " + std::to_string(cpu1[0] - cpu0[0]) + "/" +
                               std::to_string(cpu1[1] - cpu0[1]) + " for " + std::to_string(calls) + " calls of " +
                               std::to_string(k) + " lanes");
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
        if (bench.bare_p50_us.contains(k)) state.counters["forward_vs_bare_p50_us"] = forward_p50 - bench.bare_p50_us[k];
        state.counters["handoff_p50_us"] = quantile_us(handoff, 0.50);
        state.counters["handoff_p95_us"] = quantile_us(handoff, 0.95);
      }
    }
    state.counters["experts"] = k;
    state.counters["workers"] = static_cast<double>(bench.placement().workers.size());
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
    if (sglang_exl3_cpu_experts_set_cores(placement.workers.data(), static_cast<int32_t>(placement.workers.size())) != 0)
      throw std::runtime_error("Cannot configure kernel cores");
    const auto before = task_ids();
    std::unique_ptr<StackFixture> fixture;
    {
      PinScope first_touch(placement.workers.front());  // slabs, x and outputs on the workers' node
      fixture = std::make_unique<StackFixture>(options.fixture, options.image_dir);
    }
    pin_self(placement.writer);  // the writer's stores and clock; the stack's page and lease are first-touched here
    LayerHandles layers;  // outlives the Bench, whose stack's CPU expert thread uses them
    for (int64_t row = 0; row < fixture->rows(); ++row) {
      fixture->preload_slots(row);
      layers.handles.push_back(fixture->register_layer(row));
    }
    Bench bench(options, placement, *fixture, layers.handles, before);
    const std::vector<int> expected = expected_threads(placement);
    if (options.validate_only) {
      {
        PinScope caller(placement.workers.front());  // the kernel runs its caller as worker 0, on that core
        for (int k : {1, 3, 5}) bench.validate(k, /*via_stack=*/false);
        bench.check_threads();
      }
      for (int k : {1, 3, 5}) bench.validate(k, /*via_stack=*/true);  // the first builds the stack
      bench.check_threads();
      std::cerr << "Verified 48 bit-exact layer outputs (24 through the stack, 24 bare); threads pinned to {"
                << cpu_list(expected) << "}; writer CPU " << placement.writer << " (" << BenchBuild::kName
                << " build)\n";
      return 0;
    }
    benchmark::AddCustomContext("backend", EXL3_BENCH_BACKEND);
    benchmark::AddCustomContext("host_build", std::string(BenchBuild::kName));
    benchmark::AddCustomContext("fixture", options.fixture.string());
    benchmark::AddCustomContext("image_dir", options.image_dir.string());
    benchmark::AddCustomContext("writer_cpu", std::to_string(placement.writer));
    benchmark::AddCustomContext("service_cpu", std::to_string(placement.service));
    benchmark::AddCustomContext("copy_cpu", std::to_string(placement.copy));
    benchmark::AddCustomContext("worker_cpus", cpu_list(std::vector<int>(placement.workers.begin(), placement.workers.end())));
    std::ifstream cgroup_file("/proc/self/cgroup");
    benchmark::AddCustomContext("cgroup", std::string((std::istreambuf_iterator<char>(cgroup_file)), {}));
    benchmark::AddCustomContext("gap_us", std::to_string(options.gap_us));
    benchmark::AddCustomContext("compiler", __VERSION__);
    // Every BM_bare before any BM_stack: one OpenMP team at a time (Bench::enter_stack_phase). Each benchmark checks
    // its path's 8 outputs bit-exactly before and after it is timed.
    for (int k : {1, 3, 5})
      benchmark::RegisterBenchmark("BM_bare/experts:" + std::to_string(k),
                                   [&bench, k](benchmark::State& state) { bm_bare(state, bench, k); })
          ->UseManualTime()
          ->Unit(benchmark::kMicrosecond);
    for (int k : {1, 3, 5})
      benchmark::RegisterBenchmark("BM_stack/experts:" + std::to_string(k),
                                   [&bench, k](benchmark::State& state) { bm_stack(state, bench, k); })
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
