// The bare CPU-expert forward benchmark: one full forward through the kernel interface, per backend (the baseline
// through its own entry point).
//
// Built twice (EXL3_BENCH_BACKEND): the baseline kernel and the optimized one run as separate processes, so one
// backend's idle workers cannot disturb the other's timings. A forward is timed from the forward call alone; sample
// storage and layer rotation are outside the interval. Before and after each timed run, all eight layers' outputs are
// compared bit-exactly with the frozen references; setup, warmup and checks are excluded from the reported latency.
//
//   Options / parse_options   the command line
//   Workload                  one expert count's layers, slots and routing weights
//   full_forward              the Google Benchmark body and its p50/p95/p99 counters
//
// See python/sglang/kernels/jit/csrc/moe/expert_stream/bench/README.txt.
#include <ATen/Parallel.h>
#include <benchmark/benchmark.h>

#include "fixture.h"
#include "kernel.h"
#include "moe_mul1.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sched.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#ifdef EXL3_BENCH_BASELINE
#include <c10/util/Half.h>
#include <span>
// The vendored baseline's entry points (csrc/exl3/moe_mul1.cpp, its end); no header declares them.
void exl3_moe_cpu_baseline_forward(int64_t handle, const at::Half* x, const int32_t* slots, const float* weights,
                                   float* out, int rows, int k, int threads, bool accumulate);
void exl3_moe_cpu_baseline_set_cores(const int* cores, int n);
#endif

namespace {
namespace fs = std::filesystem;
std::vector<int> g_cores;  // the bench's worker cores, worker 0 first
bool benchmark_failed = false;
// The command line; the defaults are the reference machine's.
struct Options {
  fs::path fixture = "/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin";
  fs::path references = "/data/models/exl3_exp/threading";
  std::string cpus = "18-33";
  int workers = EXL3_BENCH_DEFAULT_WORKERS;
  int node = 1;
  int warmup = 128;
  int gap_us = 0;
  bool validate_only = false;
};

// Parses a non-negative decimal integer; throws on anything else.
int number(const std::string& text) {
  size_t end;
  const int n = std::stoi(text, &end);
  if (end != text.size() || n < 0) throw std::runtime_error("Invalid integer: " + text);
  return n;
}

// Parses a CPU list such as "18-33,52": ranges and singles, no duplicates. Throws on an invalid list.
std::vector<int32_t> parse_cpus(const std::string& text) {
  std::vector<int32_t> cores;
  std::stringstream list(text);
  std::string item;
  while (std::getline(list, item, ',')) {
    const auto dash = item.find('-');
    const int first = number(item.substr(0, dash));
    const int last = dash == std::string::npos ? first : number(item.substr(dash + 1));
    if (first > last || last >= CPU_SETSIZE) throw std::runtime_error("Invalid CPU range");
    for (int cpu = first; cpu <= last; ++cpu) {
      if (std::find(cores.begin(), cores.end(), cpu) != cores.end()) throw std::runtime_error("Duplicate CPU");
      cores.push_back(cpu);
    }
  }
  if (cores.empty()) throw std::runtime_error("Empty CPU list");
  return cores;
}

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
    else if (arg.starts_with("--cpus="))
      opt.cpus = value("--cpus=");
    else if (arg.starts_with("--workers="))
      opt.workers = number(value("--workers="));
    else if (arg.starts_with("--numa-node="))
      opt.node = number(value("--numa-node="));
    else if (arg.starts_with("--warmup-forwards="))
      opt.warmup = number(value("--warmup-forwards="));
    else if (arg.starts_with("--gap-us="))
      opt.gap_us = number(value("--gap-us="));
    else if (arg == "--validate-only")
      opt.validate_only = true;
    else if (arg == "--help") {
      std::cout << "Native DSV4.1 full-forward benchmark (" EXL3_BENCH_BACKEND
                   ")\n"
                   "--fixture=FILE --reference-dir=DIR --cpus=18-33 --workers=N\n"
                   "--numa-node=1 verifies CPU topology only; does not bind memory.\n"
                   "--warmup-forwards=128 --gap-us=0 --validate-only\n"
                   "Google Benchmark flags are also accepted.\n";
      argv[remaining++] = argv[i];
    } else
      argv[remaining++] = argv[i];
  }
  argc = remaining;
  argv[remaining] = nullptr;
  if (opt.workers < 1 || opt.workers > CPU_SETSIZE) throw std::runtime_error("Invalid worker count");
  return opt;
}

// The ids of this process's threads.
std::set<int> task_ids() {
  std::set<int> result;
  for (const auto& entry : fs::directory_iterator("/proc/self/task"))
    result.insert(number(entry.path().filename().string()));
  return result;
}

// Throws unless every thread created since `before` (plus the caller) is pinned to exactly one CPU and those CPUs equal
// `cores`.
void verify_workers(const std::set<int>& before, const std::vector<int32_t>& cores) {
  std::vector<int> assigned;
  for (int tid : task_ids()) {
    if (tid != getpid() && before.contains(tid)) continue;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(tid, sizeof(mask), &mask) || CPU_COUNT(&mask) != 1)
      throw std::runtime_error("Worker is not individually pinned");
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &mask)) assigned.push_back(cpu);
  }
  auto expected = cores;
  std::sort(expected.begin(), expected.end());
  std::sort(assigned.begin(), assigned.end());
  if (assigned != expected) throw std::runtime_error("Worker count/CPU assignment mismatch");
}

#ifndef EXL3_BENCH_BASELINE
// One fixture layer's first `experts` experts as the pinned tier holds them, expert e in slot e of six slabs
// (w13_trellis, w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh; w13 rows hold gate then up), and the kernel's layer
// over them.
struct SlabLayer {
  struct Free {
    void operator()(uint8_t* p) const {
      std::free(p);
    }
  };
  std::array<std::unique_ptr<uint8_t, Free>, 6> slabs;
  ::sglang::cpu_experts::ExpertLayer layer;

  SlabLayer(const LayerFixture& f, int hidden, int experts) {
    ::sglang::cpu_experts::ExpertLayer shape;
    shape.capacity = experts;
    shape.hidden = hidden;
    shape.intermediate = static_cast<int32_t>(f.matrices[0][0].size(1)) * 16;
    shape.activation = 0;
    shape.act_limit = 10.0f;
    shape.slab_count = 6;
    for (int n = 0; n < 6; ++n) {
      // The matrices (fixture.h's order) one slot's row of slab n holds: gate's then up's for w13, down's for w2.
      std::vector<const std::vector<at::Tensor>*> parts;
      if (n < 3) parts = {&f.matrices[n], &f.matrices[3 + n]};
      else parts = {&f.matrices[3 + n]};
      size_t row = 0;
      for (const auto* part : parts) row += (*part)[0].nbytes();
      slabs[n].reset(static_cast<uint8_t*>(std::aligned_alloc(64, (experts * row + 63) / 64 * 64)));
      if (!slabs[n]) throw std::runtime_error("Cannot allocate a slab");
      for (int e = 0; e < experts; ++e) {
        uint8_t* dst = slabs[n].get() + e * row;
        for (const auto* part : parts) {
          const at::Tensor t = (*part)[e].contiguous();
          std::memcpy(dst, t.data_ptr(), t.nbytes());
          dst += t.nbytes();
        }
      }
      shape.slabs[n] = slabs[n].get();
      shape.slot_bytes[n] = row;
    }
    const SglangExl3CpuParams params{3, 0};  // bits, swizzled
    layer = ::sglang::exl3_cpu::exl3_cpu_kernel().make_layer(
        shape, std::as_bytes(std::span<const SglangExl3CpuParams>(&params, 1)));
  }
};
#endif

// One expert count k: every layer registered with the first k experts, expert e in slot e, and fixed routing weights
// (the coefficients the frozen references were computed with). Owns the layers: the baseline's upstream handles, or
// the optimized kernel's slab layers.
struct Workload {
  const Fixture& fixture;
  const Options& options;
  const int experts;
#ifdef EXL3_BENCH_BASELINE
  std::vector<int64_t> handles;
#else
  std::vector<std::unique_ptr<SlabLayer>> layers;
#endif
  std::vector<int32_t> slots;
  std::vector<float> weights;
  std::vector<float> output;

  Workload(const Fixture& f, const Options& opt, int e) : fixture(f), options(opt), experts(e), output(f.hidden) {
#ifdef EXL3_BENCH_BASELINE
    try {
      for (const auto& layer : f.layers) {
        std::array<std::vector<at::Tensor>, 9> matrices;
        for (int m = 0; m < 9; ++m)
          matrices[m].assign(layer.matrices[m].begin(), layer.matrices[m].begin() + e);
        handles.push_back(exl3_moe_cpu_make_layer(
            matrices[0],
            matrices[1],
            matrices[2],
            matrices[3],
            matrices[4],
            matrices[5],
            matrices[6],
            matrices[7],
            matrices[8],
            {},
            {},
            {},
            0,
            10.0,
            0));
      }
    } catch (...) {
      for (auto handle : handles)
        exl3_moe_cpu_free_layer(handle);
      throw;
    }
#else
    for (const auto& layer : f.layers)
      layers.push_back(std::make_unique<SlabLayer>(layer, f.hidden, e));
#endif
    for (int i = 0; i < experts; ++i) {
      slots.push_back(i);
      weights.push_back(0.071234f + (experts == 1 ? 0.0f : 0.23f * i / (experts - 1)));
    }
  }
  ~Workload() {
#ifdef EXL3_BENCH_BASELINE
    for (auto handle : handles)
      exl3_moe_cpu_free_layer(handle);
#endif
  }

  size_t layer_count() const {
    return fixture.layers.size();
  }

  // One full forward on `layer`, writing `output`. Throws if the call fails.
  void forward(size_t layer) {
#ifdef EXL3_BENCH_BASELINE
    exl3_moe_cpu_baseline_forward(handles[layer], static_cast<const at::Half*>(fixture.layers[layer].input.data_ptr()),
                                  slots.data(), weights.data(), output.data(), 1, experts, options.workers, false);
#else
    ::sglang::cpu_experts::ForwardCall call;
    call.rows = 1;
    call.k = experts;
    call.threads = options.workers;
    call.x = fixture.layers[layer].input.data_ptr();
    call.slots = slots.data();
    call.weights = weights.data();
    call.out = output.data();
    call.cores = g_cores;
    ::sglang::exl3_cpu::exl3_cpu_kernel().forward(layers[layer]->layer, call);
#endif
  }

  // Runs every layer once and compares the outputs bit-exactly with the reference for this expert count. Throws on a
  // non-finite value or a mismatch.
  void validate() {
    std::vector<float> results(fixture.layers.size() * fixture.hidden);
    for (size_t layer = 0; layer < fixture.layers.size(); ++layer) {
      forward(layer);
      for (float value : output)
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite output");
      std::copy(output.begin(), output.end(), results.begin() + layer * fixture.hidden);
    }
    compare_reference(options.references / ("reference-e" + std::to_string(experts) + ".bin"), results);
  }
};

// The `fraction` quantile of `samples` in microseconds (nearest rank); reorders `samples`.
double quantile(std::vector<double>& samples, double fraction) {
  const size_t index = size_t(fraction * (samples.size() - 1));
  std::nth_element(samples.begin(), samples.begin() + index, samples.end());
  return samples[index] * 1e6;
}

// The benchmark body: validate, warm up, time `forward` per iteration, validate again. A failure marks the benchmark
// skipped with the error and fails the process.
void full_forward(benchmark::State& state, Workload& workload) {
  try {
    workload.validate();
    for (int i = 0; i < workload.options.warmup; ++i)
      workload.forward(i % workload.layer_count());
    std::vector<double> samples;
    size_t layer = 0;
    for (auto _ : state) {
      if (workload.options.gap_us) std::this_thread::sleep_for(std::chrono::microseconds(workload.options.gap_us));
      const auto start = std::chrono::steady_clock::now();
      workload.forward(layer);
      const auto end = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(end - start).count();
      state.SetIterationTime(seconds);
      // Sample storage and layer selection stay outside the timed interval.
      samples.push_back(seconds);
      layer = (layer + 1) % workload.layer_count();
      benchmark::DoNotOptimize(workload.output.data());
    }
    workload.validate();
    if (!samples.empty()) {
      state.counters["p50_us"] = quantile(samples, 0.50);
      state.counters["p95_us"] = quantile(samples, 0.95);
      state.counters["p99_us"] = quantile(samples, 0.99);
    }
    state.counters["experts"] = workload.experts;
    state.counters["workers"] = workload.options.workers;
    state.counters["layers"] = workload.layer_count();
  } catch (const std::exception& error) {
    benchmark_failed = true;
    state.SkipWithError(error.what());
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    auto cores = parse_cpus(options.cpus);
    if (options.workers > int(cores.size())) throw std::runtime_error("Not enough configured CPUs");
    cores.resize(options.workers);
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("Cannot read caller affinity");
    for (int cpu : cores) {
      if (!CPU_ISSET(cpu, &allowed))
        throw std::runtime_error("Requested CPU excluded by affinity/cgroup: " + std::to_string(cpu));
      if (!fs::exists("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(options.node)))
        throw std::runtime_error("Requested CPU is not on the expected NUMA node: " + std::to_string(cpu));
    }
    setenv("EXL3_MOE_CPU_PIN", "0", 1);
    setenv("EXL3_MOE_CPU_SMALL_WORKERS", "0", 1);
    // The tier is fixed from EXL3_MOE_CPU_MAX_ISA as set at launch: at load (baseline) or at this first query.
    if (!exl3_moe_cpu_has_avx512_bw() || exl3_moe_cpu_has_avx512_vnni() || exl3_moe_cpu_has_avx512_vbmi())
      throw std::runtime_error("This study requires AVX512BW; set EXL3_MOE_CPU_MAX_ISA=bw before launch");
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    omp_set_dynamic(0);
    // The optimized kernel takes the cores on each call (distinct, in [0, CPU_SETSIZE); one that cannot be pinned
    // fails the forward); the vendored baseline keeps its own process-wide core list.
    g_cores.assign(cores.begin(), cores.end());
#ifdef EXL3_BENCH_BASELINE
    exl3_moe_cpu_baseline_set_cores(g_cores.data(), static_cast<int>(g_cores.size()));
#endif
    cpu_set_t caller;
    CPU_ZERO(&caller);
    CPU_SET(cores.front(), &caller);
    if (sched_setaffinity(0, sizeof(caller), &caller)) throw std::runtime_error("Cannot pin caller");
    const auto before = task_ids();
    const Fixture fixture(options.fixture);
    std::vector<std::unique_ptr<Workload>> workloads;
    for (int experts : {1, 3, 5}) {
      auto workload = std::make_unique<Workload>(fixture, options, experts);
      workload->validate();
      workloads.push_back(std::move(workload));
    }
    verify_workers(before, cores);
    std::cerr << "Verified 24 bit-exact layer outputs; individually pinned worker CPUs: ";
    for (int cpu : cores)
      std::cerr << cpu << ' ';
    std::cerr << "(NUMA " << options.node << "; memory policy inherited)\n";
    if (!options.validate_only) {
      benchmark::AddCustomContext("backend", EXL3_BENCH_BACKEND);
      benchmark::AddCustomContext("fixture", options.fixture.string());
      std::string assigned_cpus;
      for (int cpu : cores)
        assigned_cpus += (assigned_cpus.empty() ? "" : ",") + std::to_string(cpu);
      benchmark::AddCustomContext("cpu_list", assigned_cpus);
      std::ifstream cgroup_file("/proc/self/cgroup");
      const std::string cgroup((std::istreambuf_iterator<char>(cgroup_file)), {});
      benchmark::AddCustomContext("cgroup", cgroup);
      benchmark::AddCustomContext("workers", std::to_string(options.workers));
      benchmark::AddCustomContext("gap_us", std::to_string(options.gap_us));
      benchmark::AddCustomContext("memory_policy", "inherited; no benchmark membind");
      benchmark::AddCustomContext("compiler", __VERSION__);
      for (auto& workload : workloads) {
        auto* w = workload.get();
        benchmark::RegisterBenchmark(
            (std::string(EXL3_BENCH_BACKEND) + "/experts:" + std::to_string(w->experts)).c_str(),
            [w](benchmark::State& state) { full_forward(state, *w); })
            ->UseManualTime()
            ->Unit(benchmark::kMicrosecond);
      }
      benchmark::RunSpecifiedBenchmarks();
      benchmark::Shutdown();
      verify_workers(before, cores);
      if (benchmark_failed) return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
