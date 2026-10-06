// Startup calibration of the CPU expert split: times the CPU lanes and the DMA of one row, alone and together.
//
// The CPU split table (RamTier::set_cpu_split) says how many of a layer's n eligible lanes the CPU computes while the
// rest cross PCIe. This file measures, on the loaded model and this machine, the mean time of k CPU lanes, of m DMA'd
// experts, and of both started together, so the split can be chosen from measured costs instead of constants. The
// combined measurement is the one that decides the split: it includes the CPU kernel and the DMA contending for host
// memory bandwidth, which the two single measurements cannot show.
//
// It runs once, on the tier's owner (a caller that paused the service thread), while the copy engine is not armed, so
// the device types no CPU or copy-engine lanes and nothing competes with the measurement. Nothing here runs on the
// decode path.
//
// The functions are free of tier state: RamTier::calibrate_cpu_split validates the row, builds a CalibrationSetup and
// calls calibrate_split().
#pragma once

#include "copy_engine.h"
#include "cpu_experts.h"
#include "host_copy_backend.h"
#include "reader_base.h"
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::expert_stream {

// The most lanes measured: one expert per lease lane.
constexpr int kCalibLanes = static_cast<int>(wire::Wire::kLanes);
// The result grid is float64, row-major [kCalibRows][kCalibCols], in ms:
//   row 0       cpu[k]      k CPU lanes alone
//   row 1       link[m]     m DMA'd experts alone
//   row 1 + n   both[n][k]  k CPU lanes and n - k DMA'd experts together, for k <= n
// Column 0 of rows 0 and 1, and the cells with k > n, stay 0.
constexpr int kCalibCols = kCalibLanes + 1;
constexpr int kCalibRows = kCalibLanes + 2;

// Everything one calibration needs, built by RamTier::calibrate_cpu_split.
//
// All measurements use one row and its host slots first_slot..first_slot + kCalibLanes - 1. Slot contents do not matter
// for timing; the bytes are real pinned memory of the real size and format. The DMA goes through `backend`, a private
// CopyBackend with its own stream and completion word, and lands in `scratch`, so calibration touches no destination
// tensor and no CopyJob.
struct CalibrationSetup {
  CpuExpertEngine* cpu = nullptr;
  int64_t row = 0;
  int64_t first_slot = 0;  // the first of the kCalibLanes host slots measured (a NUMA group's lowest)
  std::vector<CopyEntry> entries;  // the entries production's DMA copies
  CopyBackend* backend = nullptr;
  HostCopyBackend* host_backend = nullptr;  // set: release each mark (the test backend completes only released ones)
  uint64_t scratch = 0;                     // `lanes` experts: entry e's rows at its own offset
  int lanes = kCalibLanes;  // the most lanes measured, 1..kCalibLanes (spill: the victim lanes)
  int reps = 1;                             // timed repetitions per cell, after one discarded warm-up
  int64_t timeout_ns = 0;                   // per measurement
};

// The bytes of one expert that the DMA moves: the sum over the copy-table entries.
inline int64_t calibration_expert_bytes(std::span<const CopyEntry> entries) {
  int64_t bytes = 0;
  for (const CopyEntry& entry : entries)
    bytes += entry.bytes;
  return bytes;
}

// Runs k CPU lanes on host slots first_slot.. first_slot + k - 1 and the DMA of m experts from the k slots after them,
// started together, and returns the ns from the start until both are observed done (queueing and wake-up included).
// Either side may be empty. Throws on a failed copy, a full CPU ring, or past the timeout.
inline int64_t calibration_run(const CalibrationSetup& s, int k, int m) {
  const int64_t start = now_ns();
  uint32_t seq = 0;
  if (k > 0) {
    CpuJob job;
    job.row = s.row;
    job.part = 0;
    job.k = k;
    for (int i = 0; i < k; ++i) {
      job.slots[i] = static_cast<int32_t>(s.first_slot + i);
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
        const uint64_t src = entry.src + static_cast<uint64_t>(s.first_slot + k + j) * bytes;
        if (const int r = s.backend->issue(base + static_cast<uint64_t>(j) * bytes, src, entry.bytes))
          throw std::runtime_error("calibration: a DMA issue failed (" + std::to_string(r) + ")");
      }
      base += static_cast<uint64_t>(s.lanes) * bytes;
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
      throw std::runtime_error(
          "calibration: a measurement did not finish within " + std::to_string(s.timeout_ns / 1'000'000) + " ms");
    _mm_pause();
  }
  return end - start;
}

// The mean of s.reps runs of calibration_run, in ms.
inline double calibration_mean_ms(const CalibrationSetup& s, int k, int m) {
  calibration_run(s, k, m);  // warm-up: page faults, the engine's wake from its futex sleep
  int64_t sum = 0;
  for (int r = 0; r < s.reps; ++r)
    sum += calibration_run(s, k, m);
  return static_cast<double>(sum) / 1e6 / s.reps;
}

// Fills `out` (kCalibRows x kCalibCols doubles, layout above) with the mean ms of every cell.
inline void calibrate_split(const CalibrationSetup& s, double* out) {
  std::fill(out, out + kCalibRows * kCalibCols, 0.0);
  for (int k = 1; k <= s.lanes; ++k)
    out[k] = calibration_mean_ms(s, k, 0);
  for (int m = 1; m <= s.lanes; ++m)
    out[kCalibCols + m] = calibration_mean_ms(s, 0, m);
  for (int n = 1; n <= s.lanes; ++n)
    for (int k = 0; k <= n; ++k)
      out[(1 + n) * kCalibCols + k] = calibration_mean_ms(s, k, n - k);
}

}  // namespace sglang::expert_stream
