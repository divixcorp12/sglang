// The full-stack CPU-expert benchmark (docs/superpowers/specs/2026-10-01-expert-stream-full-stack-bench-design.md):
// a writer thread posts CPU-hit requests into the lease lanes; the real RamTier, RamThread and CpuExpertEngine serve
// them with the optimized EXL3 kernel; the writer times post -> CopyDone against the bare kernel call.
#include <benchmark/benchmark.h>

#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "placement.h"
#include "self_test.h"

namespace {
namespace fs = std::filesystem;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    const Placement placement = resolve_placement(options);
    validate_placement(placement, system_topology(), !options.self_test);
    if (options.self_test) return run_self_test(placement, options.image_dir) == 0 ? 0 : 1;
    throw std::runtime_error("only --self-test is built yet");
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
