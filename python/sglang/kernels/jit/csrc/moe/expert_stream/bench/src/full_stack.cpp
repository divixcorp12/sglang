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
  Bench(const Options& options, Placement placement, StackFixture& fixture, Stack<BenchBuild>& stack, DeviceSim& sim,
        std::vector<int64_t> handles)
      : options_(options),
        placement_(std::move(placement)),
        fixture_(fixture),
        stack_(stack),
        sim_(sim),
        handles_(std::move(handles)),
        deadline_ns_(int64_t{options.wait_timeout_ms} * 1'000'000 / 2) {
    for (int k : {1, 3, 5}) {
      for (int i = 0; i < k; ++i) {
        experts_[k].push_back(i);
        // cpu_forward.cpp's routing coefficients, which the frozen references were computed with
        weights_[k].push_back(0.071234f + (k == 1 ? 0.0f : 0.23f * i / (k - 1)));
      }
      for (int64_t row = 0; row < fixture.rows(); ++row) {
        std::vector<int32_t> slots;
        for (int32_t e : experts_[k]) slots.push_back(sim.ram_slot(row, e));
        slots_[k].push_back(std::move(slots));
      }
    }
  }

  const Options& options() const { return options_; }
  const Placement& placement() const { return placement_; }
  Stack<BenchBuild>& stack() { return stack_; }
  int64_t rows() const { return fixture_.rows(); }

  // The device's part of one request: x, the record, the copy wait. t0..t1 is the timed interval.
  StackCall stack_call(int64_t row, int k) {
    StackCall call;
    call.t0 = monotonic_ns();
    fixture_.write_x(row);
    call.request = sim_.post(row, experts_[k], weights_[k], /*captured=*/true, call.t0 + deadline_ns_);
    const bool done = sim_.copy_wait(call.request, call.t0 + deadline_ns_);
    call.t1 = monotonic_ns();
    if (!done) throw std::runtime_error("the copy wait passed its deadline: " + describe(call.request));
    for (int j = 0; j < k; ++j) {
      if (call.request.kinds[j] != static_cast<int32_t>(es::kKindHitCpu))
        throw std::runtime_error("lane " + std::to_string(j) + " was not typed HIT_CPU: " + describe(call.request));
    }
    return call;
  }

  // The C ABI forward on the same layer handle, slots, x and output memory, from the calling thread.
  void bare_call(int64_t row, int k) {
    if (sglang_exl3_cpu_experts_forward(handles_[row], fixture_.x_row(row), slots_[k][row].data(), weights_[k].data(), k,
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
    const auto c = stack_.counters();
    const auto cpu = stack_.cpu_stats();
    s << "; served " << c[es::kServedRequests] << " touch_only " << c[es::kTouchOnly] << " rows_read "
      << c[es::kRowsRead] << " overruns " << c[es::kOverruns] << "; cpu jobs " << cpu[0] << " lanes " << cpu[1]
      << "; CopyDone " << sim_.copy_done(r) << ", gate 0x" << std::hex << sim_.copy_gate() << std::dec
      << ", handled through " << stack_.tier().handled_through();
    return s.str();
  }

  std::map<int, double> bare_p50_us;  // BM_bare's p50 per k, for BM_stack's overhead counter

 private:
  const Options& options_;
  Placement placement_;
  StackFixture& fixture_;
  Stack<BenchBuild>& stack_;
  DeviceSim& sim_;
  std::vector<int64_t> handles_;
  int64_t deadline_ns_;
  std::map<int, std::vector<int32_t>> experts_;
  std::map<int, std::vector<float>> weights_;
  std::map<int, std::vector<std::vector<int32_t>>> slots_;  // [k][row]
};

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
    pin_self(placement.writer);  // the writer's stores and clock; the page and lease are first-touched here
    LayerHandles layers;
    Stack<BenchBuild> stack(stack_config(*fixture, placement, options));
    DeviceSim sim(stack.page(), stack.lease(), fixture->rows(), fixture->experts());
    const int64_t timeout_ns = int64_t{options.wait_timeout_ms} * 1'000'000;
    std::vector<int32_t> all(static_cast<size_t>(fixture->experts()));
    std::iota(all.begin(), all.end(), 0);
    for (int64_t row = 0; row < fixture->rows(); ++row) {
      load_experts(sim, row, all, static_cast<int>(StackFixture::kStaging), timeout_ns);
      layers.handles.push_back(fixture->register_layer(row));
      stack.set_cpu_layer(row, layers.handles.back());
      sim.set_row_cpu(row);
    }
    Bench bench(options, placement, *fixture, stack, sim, layers.handles);
    for (int k : {1, 3, 5}) {
      bench.validate(k, /*via_stack=*/true);
      PinScope caller(placement.workers.front());  // the kernel's caller is worker 0, as on the CPU expert thread
      std::this_thread::sleep_for(std::chrono::milliseconds(100));  // that thread's 50 ms idle spin ends first
      bench.validate(k, /*via_stack=*/false);
    }
    const std::vector<int> expected = expected_threads(placement);
    verify_threads(before, expected);
    std::cerr << "Verified 48 bit-exact layer outputs (24 through the stack, 24 bare); threads pinned to {"
              << cpu_list(expected) << "}; writer CPU " << placement.writer << " (" << BenchBuild::kName << " build)\n";
    if (!options.validate_only) throw std::runtime_error("timing is not built yet: run with --validate-only");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
