#include "fixture.h"
#include "moe_mul1.h"
#include "cpu_experts_cabi.h"
#include <ATen/Parallel.h>
#include <benchmark/benchmark.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
namespace fs = std::filesystem;
bool benchmark_failed = false;
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

int number(const std::string& text) {
  size_t end;
  const int n = std::stoi(text, &end);
  if (end != text.size() || n < 0) throw std::runtime_error("Invalid integer: " + text);
  return n;
}

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

Options parse_options(int& argc, char** argv) {
  Options opt;
  int remaining = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value = [&](const std::string& prefix) { return arg.substr(prefix.size()); };
    if (arg.starts_with("--fixture=")) opt.fixture = value("--fixture=");
    else if (arg.starts_with("--reference-dir=")) opt.references = value("--reference-dir=");
    else if (arg.starts_with("--cpus=")) opt.cpus = value("--cpus=");
    else if (arg.starts_with("--workers=")) opt.workers = number(value("--workers="));
    else if (arg.starts_with("--numa-node=")) opt.node = number(value("--numa-node="));
    else if (arg.starts_with("--warmup-forwards=")) opt.warmup = number(value("--warmup-forwards="));
    else if (arg.starts_with("--gap-us=")) opt.gap_us = number(value("--gap-us="));
    else if (arg == "--validate-only") opt.validate_only = true;
    else if (arg == "--help") {
      std::cout << "Native DSV4.1 full-forward benchmark (" EXL3_BENCH_BACKEND ")\n"
        "--fixture=FILE --reference-dir=DIR --cpus=18-33 --workers=N\n"
        "--numa-node=1 verifies CPU topology only; does not bind memory.\n"
        "--warmup-forwards=128 --gap-us=0 --validate-only\n"
        "Google Benchmark flags are also accepted.\n";
      argv[remaining++] = argv[i];
    } else argv[remaining++] = argv[i];
  }
  argc = remaining;
  argv[remaining] = nullptr;
  if (opt.workers < 1 || opt.workers > CPU_SETSIZE) throw std::runtime_error("Invalid worker count");
  return opt;
}

std::set<int> task_ids() {
  std::set<int> result;
  for (const auto& entry : fs::directory_iterator("/proc/self/task"))
    result.insert(number(entry.path().filename().string()));
  return result;
}

void verify_workers(const std::set<int>& before, const std::vector<int32_t>& cores) {
  std::vector<int> assigned;
  for (int tid : task_ids()) {
    if (tid != getpid() && before.contains(tid)) continue;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(tid, sizeof(mask), &mask) || CPU_COUNT(&mask) != 1)
      throw std::runtime_error("Worker is not individually pinned");
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &mask)) assigned.push_back(cpu);
  }
  auto expected = cores;
  std::sort(expected.begin(), expected.end());
  std::sort(assigned.begin(), assigned.end());
  if (assigned != expected) throw std::runtime_error("Worker count/CPU assignment mismatch");
}

struct Workload {
  const Fixture& fixture;
  const Options& options;
  const int experts;
  std::vector<int64_t> handles;
  std::vector<int32_t> slots;
  std::vector<float> weights;
  std::vector<float> output;

  Workload(const Fixture& f, const Options& opt, int e) : fixture(f), options(opt), experts(e), output(f.hidden) {
    try {
      for (const auto& layer : f.layers) {
        std::array<std::vector<at::Tensor>, 9> matrices;
        for (int m = 0; m < 9; ++m)
          matrices[m].assign(layer.matrices[m].begin(), layer.matrices[m].begin() + e);
        handles.push_back(exl3_moe_cpu_make_layer(matrices[0], matrices[1], matrices[2],
          matrices[3], matrices[4], matrices[5], matrices[6], matrices[7], matrices[8], {}, {}, {}, 0, 10.0, 0));
      }
    } catch (...) {
      for (auto handle : handles) exl3_moe_cpu_free_layer(handle);
      throw;
    }
    for (int i = 0; i < experts; ++i) {
      slots.push_back(i);
      weights.push_back(0.071234f + (experts == 1 ? 0.0f : 0.23f * i / (experts - 1)));
    }
  }
  ~Workload() { for (auto handle : handles) exl3_moe_cpu_free_layer(handle); }

  void forward(size_t layer) {
    if (sglang_exl3_cpu_experts_forward(handles[layer], fixture.layers[layer].input.data_ptr(),
        slots.data(), weights.data(), experts, output.data(), options.workers))
      throw std::runtime_error("Native CPU forward failed");
  }

  void validate() {
    std::vector<float> results(fixture.layers.size() * fixture.hidden);
    for (size_t layer = 0; layer < fixture.layers.size(); ++layer) {
      forward(layer);
      for (float value : output) if (!std::isfinite(value)) throw std::runtime_error("Non-finite output");
      std::copy(output.begin(), output.end(), results.begin() + layer * fixture.hidden);
    }
    compare_reference(options.references / ("reference-e" + std::to_string(experts) + ".bin"), results);
  }
};

double quantile(std::vector<double>& samples, double fraction) {
  const size_t index = size_t(fraction * (samples.size() - 1));
  std::nth_element(samples.begin(), samples.begin() + index, samples.end());
  return samples[index] * 1e6;
}

void full_forward(benchmark::State& state, Workload& workload) {
  try {
    workload.validate();
    for (int i = 0; i < workload.options.warmup; ++i) workload.forward(i % workload.handles.size());
    std::vector<double> samples;
    size_t layer = 0;
    for (auto _ : state) {
      if (workload.options.gap_us)
        std::this_thread::sleep_for(std::chrono::microseconds(workload.options.gap_us));
      const auto start = std::chrono::steady_clock::now();
      workload.forward(layer);
      const auto end = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(end - start).count();
      state.SetIterationTime(seconds);
      // Framework bookkeeping, sample storage and layer selection are excluded.
      samples.push_back(seconds);
      layer = (layer + 1) % workload.handles.size();
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
    state.counters["layers"] = workload.handles.size();
  } catch (const std::exception& error) {
    benchmark_failed = true;
    state.SkipWithError(error.what());
  }
}
} // namespace

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
      if (!CPU_ISSET(cpu, &allowed)) throw std::runtime_error("Requested CPU excluded by affinity/cgroup: " + std::to_string(cpu));
      if (!fs::exists("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(options.node)))
        throw std::runtime_error("Requested CPU is not on the expected NUMA node: " + std::to_string(cpu));
    }
    setenv("EXL3_MOE_CPU_PIN", "0", 1);
    setenv("EXL3_MOE_CPU_SMALL_WORKERS", "0", 1);
    // ISA detection occurs during kernel static initialization, before main.
    if (!exl3_moe_cpu_has_avx512_bw() || exl3_moe_cpu_has_avx512_vnni() || exl3_moe_cpu_has_avx512_vbmi())
      throw std::runtime_error("This study requires AVX512BW; set EXL3_MOE_CPU_MAX_ISA=bw before launch");
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    omp_set_dynamic(0);
    cpu_set_t caller;
    CPU_ZERO(&caller);
    CPU_SET(cores.front(), &caller);
    if (sched_setaffinity(0, sizeof(caller), &caller)) throw std::runtime_error("Cannot pin caller");
    if (sglang_exl3_cpu_experts_set_cores(cores.data(), cores.size())) throw std::runtime_error("Cannot configure kernel cores");
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
    for (int cpu : cores) std::cerr << cpu << ' ';
    std::cerr << "(NUMA " << options.node << "; memory policy inherited)\n";
    if (!options.validate_only) {
      benchmark::AddCustomContext("backend", EXL3_BENCH_BACKEND);
      benchmark::AddCustomContext("fixture", options.fixture.string());
      std::string assigned_cpus;
      for (int cpu : cores) assigned_cpus += (assigned_cpus.empty() ? "" : ",") + std::to_string(cpu);
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
        benchmark::RegisterBenchmark((std::string(EXL3_BENCH_BACKEND) + "/experts:" + std::to_string(w->experts)).c_str(),
          [w](benchmark::State& state) { full_forward(state, *w); })
          ->UseManualTime()->Unit(benchmark::kMicrosecond);
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
