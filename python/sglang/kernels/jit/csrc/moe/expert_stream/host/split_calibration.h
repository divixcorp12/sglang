// Startup calibration of the CPU expert split (spec docs/superpowers/specs/2026-10-01-cpu-split-calibration-design.md):
// mean ms of k CPU lanes, of m DMA'd experts, and of both started together, on one row. Run by the tier's owner while
// the copy engine is not armed, so no device lane competes. Nothing here runs on the decode path.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "copy_engine.h"
#include "cpu_experts.h"
#include "reader_base.h"

namespace sglang::expert_stream {

constexpr int kCalibLanes = static_cast<int>(wire::kLeaseLanes);
// The grid, float64 row-major [kCalibRows][kCalibCols] in ms: row 0 cpu[k], row 1 link[m], row 1 + n both[n][k] for
// k <= n. Column 0 of rows 0 and 1, and cells k > n, stay 0.
constexpr int kCalibCols = kCalibLanes + 1;
constexpr int kCalibRows = kCalibLanes + 2;

struct CalibrationSetup {
  CpuExpertEngine* cpu = nullptr;
  int64_t row = 0;
  std::vector<CopyEntry> entries;  // the entries production's DMA copies
  CopyBackend* backend = nullptr;
  HostCopyBackend* host_backend = nullptr;  // set: release each mark (the test backend completes only released ones)
  uint64_t scratch = 0;                     // kCalibLanes experts: entry e's rows at its own offset
  int reps = 1;
  int64_t timeout_ns = 0;
};

inline int64_t calibration_expert_bytes(std::span<const CopyEntry> entries) {
  int64_t bytes = 0;
  for (const CopyEntry& entry : entries)
    bytes += entry.bytes;
  return bytes;
}

// k CPU lanes on host slots 0..k-1 and the DMA of m experts from slots k..k+m-1, started together. Returns the ns from
// the start until both are observed done. Throws on a failed copy or past the timeout.
inline int64_t calibration_run(const CalibrationSetup& s, int k, int m) {
  const int64_t start = now_ns();
  uint32_t seq = 0;
  if (k > 0) {
    CpuJob job;
    job.row = s.row;
    job.part = 0;
    job.k = k;
    for (int i = 0; i < k; ++i) {
      job.slots[i] = i;
      job.weights[i] = 1.0f;
    }
    job.seq = seq = s.cpu->claim(1);
    if (!s.cpu->submit(job)) throw std::runtime_error("calibration: the CPU expert ring is full");
  }
  int64_t token = -1;
  if (m > 0) {
    uint64_t base = s.scratch;
    for (const CopyEntry& entry : s.entries) {
      const uint64_t bytes = static_cast<uint64_t>(entry.bytes);
      for (int j = 0; j < m; ++j) {
        const uint64_t src = entry.src + static_cast<uint64_t>(k + j) * bytes;
        if (const int r = s.backend->issue(base + static_cast<uint64_t>(j) * bytes, src, entry.bytes))
          throw std::runtime_error("calibration: a DMA issue failed (" + std::to_string(r) + ")");
      }
      base += static_cast<uint64_t>(kCalibLanes) * bytes;
    }
    if (const int r = s.backend->mark(&token))
      throw std::runtime_error("calibration: a DMA mark failed (" + std::to_string(r) + ")");
    if (s.host_backend != nullptr) s.host_backend->release(-1);
  }
  bool cpu_done = k == 0;
  bool dma_done = m == 0;
  int64_t end = start;
  while (!cpu_done || !dma_done) {
    if (!cpu_done && s.cpu->done(seq)) {
      cpu_done = true;
      end = std::max(end, now_ns());
    }
    if (!dma_done) {
      const int state = s.backend->query(token);
      if (state == CopyBackend::kDone) {
        dma_done = true;
        end = std::max(end, now_ns());
      } else if (state != CopyBackend::kPending) {
        throw std::runtime_error("calibration: a DMA failed (" + std::to_string(state) + ")");
      }
    }
    if (now_ns() - start > s.timeout_ns)
      throw std::runtime_error("calibration: a measurement did not finish within " +
                               std::to_string(s.timeout_ns / 1'000'000) + " ms");
    _mm_pause();
  }
  return end - start;
}

inline double calibration_mean_ms(const CalibrationSetup& s, int k, int m) {
  calibration_run(s, k, m);  // warm-up: page faults, the engine's wake from its futex sleep
  int64_t sum = 0;
  for (int r = 0; r < s.reps; ++r)
    sum += calibration_run(s, k, m);
  return static_cast<double>(sum) / 1e6 / s.reps;
}

inline void calibrate_split(const CalibrationSetup& s, double* out) {
  std::fill(out, out + kCalibRows * kCalibCols, 0.0);
  for (int k = 1; k <= kCalibLanes; ++k)
    out[k] = calibration_mean_ms(s, k, 0);
  for (int m = 1; m <= kCalibLanes; ++m)
    out[kCalibCols + m] = calibration_mean_ms(s, 0, m);
  for (int n = 1; n <= kCalibLanes; ++n)
    for (int k = 0; k <= n; ++k)
      out[(1 + n) * kCalibCols + k] = calibration_mean_ms(s, k, n - k);
}

}  // namespace sglang::expert_stream
