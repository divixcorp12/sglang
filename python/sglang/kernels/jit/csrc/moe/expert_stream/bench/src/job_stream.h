// The job-stream driver: the CPU expert engine fed the record pattern of the production DSpark trace, with several
// records in flight.
//
// A record is a hit job (part 0, 3-5 lanes) and then 1-lane miss jobs (part 1) submitted as their rows land; every second
// miss is staged and its gate opens a little after the job goes in. The driver is the tier owner: one thread claims
// sequences, submits in sequence order, opens the gates by a timer and watches done(seq). It uses only the engine's
// submit / claim / open_stage_two / done interface, so the same source runs against the engine before the job window
// and after it. See README.txt, "Job-stream bench".
#pragma once

#include "aligned.h"
#include "device_sim.h"
#include "kernel.h"
#include "stack.h"
#include "stack_fixture.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <immintrin.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fullstack {

// The record pattern and its timing; defaults are derived in the README.
struct JobStreamShape {
  int land_us = 200;   // a record's i-th miss row lands (i + 1) * land_us after its hit job was submitted
  int gate_us = 100;   // a staged miss's gate opens this long after the job is submitted
  int per_lane = 6;    // tokens routed to each lane, in a record of several tokens (the tokens x experts sharing)
  bool store_out = false;
};

// One benchmark case.
struct JobStreamCase {
  int in_flight = 1;
  int tokens = 1;
};

// One job of a record and when, from the record's first submit, the driver submits it.
struct PlannedJob {
  es::CpuJob job;
  int64_t offset_ns = 0;
};

// The result of a stream: one completion time per record, the whole stream's wall time, and (when asked) each record's
// final output row, for comparing two runs bit for bit.
struct JobStreamRun {
  std::vector<double> record_seconds;
  double wall_seconds = 0;
  std::vector<std::vector<uint8_t>> outputs;
};

// The staged inputs, outputs and engine of one case. Construct on a thread that may first-touch on the workers' node.
template <class Build>
class JobStreamRig {
 public:
  using Engine = es::BasicCpuExpertEngine<Build>;
  static constexpr int kRecordLanes = 6;  // routed experts a token has
  static constexpr int kLoadedSlots = 5;  // the fixture's slab holds experts 0-4, at slots 0-4

  JobStreamRig(
      const StackFixture& fixture,
      const std::vector<::sglang::exl3_cpu::Exl3CpuKernel::Layer>& layers,
      const std::vector<int>& cores,
      int tokens,
      const JobStreamShape& shape,
      int64_t keep_warm_ns)
      : shape_(shape), tokens_(tokens), rows_(fixture.rows()), hidden_(fixture.hidden()), layers_(rows_) {
    x_token_bytes_ = round_up(2 * hidden_, 16);
    x_stride_ = tokens_ * x_token_bytes_ +
                (tokens_ > 1 ? es::CpuTokenTable::kHeaderBytes + 4 * es::Wire::kLanes * (1 + tokens_) : 0);
    out_part_stride_ = int64_t{tokens_} * hidden_ * static_cast<int64_t>(sizeof(float));
    out_stride_ = 2 * out_part_stride_;
    x_ = aligned_zeroed(rows_ * x_stride_);
    out_ = aligned_zeroed(rows_ * out_stride_);
    for (int64_t row = 0; row < rows_; ++row) {
      const uint8_t* input = fixture.x_row(row);
      for (int t = 0; t < tokens_; ++t) {
        uint8_t* at = x_.get() + row * x_stride_ + t * x_token_bytes_;
        std::memcpy(at, input, static_cast<size_t>(2 * hidden_));
        if (t % 2 == 1)  // distinct tokens: flip the FP16 sign bits
          for (int64_t h = 0; h < hidden_; ++h)
            at[2 * h + 1] ^= 0x80;
      }
      layers_.set(row, layers[static_cast<size_t>(row)]);
    }
    typename Engine::Config c;
    c.layers = &layers_;
    c.x_base = x_.get();
    c.x_stride = x_stride_;
    c.x_token_bytes = x_token_bytes_;
    c.tokens = tokens_;
    c.out_base = out_.get();
    c.out_stride = out_stride_;
    c.out_part_stride = out_part_stride_;
    c.hidden = hidden_;
    c.threads = static_cast<int>(cores.size());
    c.cores = cores;
    c.keep_warm_ns = keep_warm_ns;
    c.check_calls = Build::kFaults;
    c.two_stage = true;
    c.store_out = shape_.store_out;
    engine_ = std::make_unique<Engine>(std::move(c), "job-stream: ", "js-engine");
    for (int64_t row = 0; row < rows_; ++row)
      engine_->write_calibration_table(row, tokens_, tokens_ > 1 ? shape_.per_lane : 0);
    engine_->start();
  }

  // Plans the next stream from record 0 again (the sequences go on).
  void restart() {
    next_record_ = 0;
  }

  // The `index`-th record's jobs, a pure function of the index: a hit job of k = 3, 4, 5 lanes and the 6 - k miss jobs.
  std::vector<PlannedJob> plan(int64_t index) const {
    const int k = 3 + static_cast<int>(index % 3);
    const int64_t row = index % rows_;
    std::vector<PlannedJob> jobs;
    for (int m = -1; m < kRecordLanes - k; ++m) {
      PlannedJob p;
      es::CpuJob& job = p.job;
      job.row = row;
      job.part = m < 0 ? 0 : 1;
      job.accumulate = m > 0;  // the first miss job initialises part 1, the later ones add to it
      job.per_token = tokens_ > 1;
      job.staged = m >= 0 && m % 2 == 1;
      job.k = m < 0 ? k : 1;
      for (int i = 0; i < job.k; ++i) {
        const int lane = m < 0 ? i : k + m;
        job.slots[i] = lane % kLoadedSlots;
        job.weights[i] = 0.071234f + 0.23f * static_cast<float>(lane) / (kRecordLanes - 1);
        job.lanes[i] = lane;
      }
      p.offset_ns = int64_t{m + 1} * shape_.land_us * 1000;
      jobs.push_back(p);
    }
    return jobs;
  }

  // `records` records with up to `in_flight` incomplete at once: the next record's hit job goes in as soon as the
  // previous record's last job has, if fewer than `in_flight` are incomplete. Each record's time runs from its first
  // submit to the observation of done() passing its last job.
  JobStreamRun run_stream(int records, int in_flight, bool keep_outputs) {
    struct Flight {
      std::vector<PlannedJob> jobs;
      size_t next = 0;
      int64_t t0 = 0;
      uint32_t last_seq = 0;
      int64_t row = 0;
    };
    JobStreamRun result;
    std::deque<Flight> flight;
    std::vector<std::pair<int64_t, uint32_t>> gates;  // when a staged job's gate opens, and its sequence
    int started = 0, completed = 0;
    const int64_t begin = monotonic_ns();
    while (completed < records) {
      const int64_t now = monotonic_ns();
      for (size_t i = 0; i < gates.size();) {
        if (now >= gates[i].first) {
          engine_->open_stage_two(gates[i].second);
          gates.erase(gates.begin() + static_cast<std::ptrdiff_t>(i));
        } else {
          ++i;
        }
      }
      while (!flight.empty() && engine_->done(flight.front().last_seq)) {
        const int64_t t1 = monotonic_ns();
        result.record_seconds.push_back(static_cast<double>(t1 - flight.front().t0) * 1e-9);
        if (keep_outputs) {
          const uint8_t* row = out_.get() + flight.front().row * out_stride_;
          result.outputs.emplace_back(row, row + out_stride_);
        }
        flight.pop_front();
        ++completed;
      }
      if (!flight.empty() && flight.back().next < flight.back().jobs.size()) {
        Flight& f = flight.back();
        if (now >= f.t0 + f.jobs[f.next].offset_ns) submit(f.jobs[f.next++].job, gates);
      } else if (started < records && static_cast<int>(flight.size()) < in_flight) {
        Flight f;
        f.jobs = plan(next_record_++);
        f.row = f.jobs.front().job.row;
        const uint32_t first = engine_->claim(static_cast<int>(f.jobs.size()));
        for (size_t i = 0; i < f.jobs.size(); ++i)
          f.jobs[i].job.seq = first + static_cast<uint32_t>(i);
        f.last_seq = f.jobs.back().job.seq;
        f.t0 = monotonic_ns();
        submit(f.jobs[0].job, gates);
        f.next = 1;
        flight.push_back(std::move(f));
        ++started;
      }
      _mm_pause();
    }
    result.wall_seconds = static_cast<double>(monotonic_ns() - begin) * 1e-9;
    return result;
  }

  // The same records one job at a time: each job is submitted with its gate open and waited for before the next.
  JobStreamRun run_serial(int records) {
    JobStreamRun result;
    for (int r = 0; r < records; ++r) {
      std::vector<PlannedJob> jobs = plan(next_record_++);
      const uint32_t first = engine_->claim(static_cast<int>(jobs.size()));
      for (size_t i = 0; i < jobs.size(); ++i) {
        es::CpuJob& job = jobs[i].job;
        job.seq = first + static_cast<uint32_t>(i);
        if (job.staged) engine_->open_stage_two(job.seq);
        if (!engine_->submit(job)) throw std::runtime_error("the engine's ring is full");
        wait_done(job.seq);
      }
      const uint8_t* row = out_.get() + jobs.front().job.row * out_stride_;
      result.outputs.emplace_back(row, row + out_stride_);
    }
    return result;
  }

 private:
  void submit(es::CpuJob& job, std::vector<std::pair<int64_t, uint32_t>>& gates) {
    if (!engine_->submit(job)) throw std::runtime_error("the engine's ring is full");
    if (job.staged) gates.emplace_back(monotonic_ns() + int64_t{shape_.gate_us} * 1000, job.seq);
  }

  void wait_done(uint32_t seq) {
    const int64_t deadline = monotonic_ns() + 30'000'000'000;
    while (!engine_->done(seq)) {
      if (monotonic_ns() > deadline) throw std::runtime_error("job " + std::to_string(seq) + " did not finish in 30 s");
      _mm_pause();
    }
  }

  JobStreamShape shape_;
  int tokens_;
  int64_t rows_, hidden_;
  int64_t x_token_bytes_ = 0, x_stride_ = 0, out_part_stride_ = 0, out_stride_ = 0;
  AlignedBuffer x_, out_;
  typename Engine::Layers layers_;
  std::unique_ptr<Engine> engine_;
  int64_t next_record_ = 0;
};

}  // namespace fullstack
