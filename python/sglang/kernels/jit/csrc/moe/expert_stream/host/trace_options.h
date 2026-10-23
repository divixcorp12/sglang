// The diagnostics options of the host (SGLANG_EXPERT_STREAM_TRACE's job, draft_trigger and gate_path members), passed
// from Python: the host reads no environment variable. Every member's default is off, as an unset env was.
//
// A module installs one TraceOptions (set_trace_options, from the expert_stream_set_trace_options export) before it
// builds any engine object; JobTrace, DraftDelayTrigger and TraceGate copy it at construction, or take one explicitly.
#pragma once
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace sglang::expert_stream {

struct TraceOptions {
  std::string job_prefix;  // trace.job.prefix: a path prefix; empty is no job trace
  int64_t job_capacity = 131072;
  bool job_resources = false;
  // trace.draft_trigger thresholds in us, 0 off: reasons 2 (arrival), 0 (pending) and 1 (forward).
  int64_t arrival_us = 0, pending_us = 0, forward_us = 0;
  std::string trigger_report_prefix;
  std::string gate_path;  // trace.gate_path: a four-byte mmap file admitting the trace; empty always admits
};

// A threshold in ns from its us option, or std::invalid_argument naming the field.
inline int64_t trigger_threshold_ns(int64_t us, const char* field) {
  if (us < 0 || us > 1000000000)
    throw std::invalid_argument(std::string("SGLANG_EXPERT_STREAM_TRACE draft_trigger.") + field +
                                " must be an integer in [0, 1000000000]");
  return us * 1000;
}

// Internal linkage: each module's translation unit owns its installed options, never merged with another library's.
namespace {
struct InstalledTraceOptions {
  std::mutex mutex;
  TraceOptions options;
};
inline InstalledTraceOptions& installed_trace_options_state() {
  static InstalledTraceOptions state;
  return state;
}

inline void set_trace_options(TraceOptions options) {
  auto& state = installed_trace_options_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.options = std::move(options);
}

inline TraceOptions installed_trace_options() {
  auto& state = installed_trace_options_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.options;
}
}  // namespace

}  // namespace sglang::expert_stream
