// The io_uring options of the expert-stream reader: SGLANG_EXPERT_STREAM_URING's members (config.py's UringConfig),
// passed from Python. The host reads no environment variable.
//
// A module installs one UringOptions (set_uring_options, from the expert_stream_set_uring_options export) before it
// builds any reader, and each reader copies it (installed_uring_options). The defaults are production's; an invalid
// value throws std::invalid_argument naming the member. UringOptions also derives the behaviors that follow from the
// combination of options (polling, wait strategy, read cuts), so the reader asks it rather than re-deriving them.
//
//   mode, queue_depth, fixed_files, read_mode, wait_mode, sq_thread_idle_ms, diagnostics, read_cuts
// The SQPOLL core is not here: each ring takes it from the thread plan (BasicUringReader::set_sq_thread_cpu).
#pragma once

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sglang::expert_stream {

// Ring setup mode (MODE): plain, completion polling (IOPOLL), kernel submission thread (SQPOLL), or both.
enum class UringMode { Default, IoPoll, SqPoll, SqPollIoPoll };

// How reads are issued (READ_MODE): ordinary reads, or fixed reads into registered buffers, with one iovec per SQE
// (Fixed) or a scattered iovec array per SQE (ReadvFixed, which needs liburing 2.10).
enum class UringReadMode { Normal, Fixed, ReadvFixed };

// How the reader waits for completions (WAIT_MODE): sleeping in the kernel, or polling the completion queue.
enum class UringWaitMode { Block, Spin };

// Whether reads are cut into legs the block device takes whole (READ_CUTS): Auto follows IOPOLL.
enum class UringReadCuts { Auto, Off, On };

// The parsed options. Build one with parse(); the const members derive the behavior the options imply.
struct UringOptions {
  unsigned queue_depth = 0;  // 0 keeps RowReader's 16 * parts credit limit
  UringMode mode = UringMode::Default;
  bool fixed_files = false;
  UringReadMode read_mode = UringReadMode::Normal;
  UringWaitMode wait_mode = UringWaitMode::Block;
  unsigned sq_thread_idle_ms = 10000;
  int sq_thread_cpu = -1;  // unpinned unless the ring's set_sq_thread_cpu names a core
  bool diagnostics = false;
  // Cut every read into legs the block device takes whole (read_cuts.h). Auto: on exactly when IOPOLL is, where an
  // uncut read is punted to io-wq (analysis/dsv41-drive/iopoll/diagnosis.md).
  UringReadCuts read_cuts = UringReadCuts::Auto;
  // True when reads are cut, after resolving Auto.
  bool read_cuts_on() const {
    return read_cuts == UringReadCuts::On || (read_cuts == UringReadCuts::Auto && iopoll());
  }
  const char* read_cuts_name() const {
    return read_cuts == UringReadCuts::Auto ? "auto" : read_cuts == UringReadCuts::On ? "on" : "off";
  }

  // True when the ring has a kernel submission thread.
  bool sqpoll() const {
    return mode == UringMode::SqPoll || mode == UringMode::SqPollIoPoll;
  }
  // True when completions are polled from the device instead of interrupt-driven.
  bool iopoll() const {
    return mode == UringMode::IoPoll || mode == UringMode::SqPollIoPoll;
  }
  // True for IOPOLL without SQPOLL. io_uring_enter(GETEVENTS, min_complete>0) polls the device inside the kernel
  // holding the ring's uring_lock, and a read punted to io-wq cannot queue itself on the poll list until the waiter
  // lets go: the punted reads then issue one after another behind completions (+1.5 ms per row;
  // analysis/dsv41-drive/iopoll/diagnosis.md table 3). Such a ring therefore always waits with min_complete=0
  // passes.
  bool polls_in_wait() const {
    return iopoll() && !sqpoll();
  }
  // True when submit() may sleep in the kernel until completions arrive.
  bool blocking_wait() const {
    return wait_mode == UringWaitMode::Block && !polls_in_wait();
  }
  const char* effective_wait_name() const {
    return blocking_wait() ? "block" : polls_in_wait() ? "reap" : "spin";
  }
  const char* mode_name() const {
    switch (mode) {
      case UringMode::Default:
        return "default";
      case UringMode::IoPoll:
        return "iopoll";
      case UringMode::SqPoll:
        return "sqpoll";
      case UringMode::SqPollIoPoll:
        return "sqpoll_iopoll";
    }
    return "invalid";
  }
  const char* read_mode_name() const {
    switch (read_mode) {
      case UringReadMode::Normal:
        return "normal";
      case UringReadMode::Fixed:
        return "fixed";
      case UringReadMode::ReadvFixed:
        return "readv_fixed";
    }
    return "invalid";
  }

  // Validates SGLANG_EXPERT_STREAM_URING's members as Python passes them. Throws std::invalid_argument on a bad value.
  static UringOptions parse(std::string_view mode, int64_t queue_depth, bool fixed_files, std::string_view read,
                            std::string_view wait, int64_t sq_thread_idle_ms, bool diagnostics, std::string_view cuts) {
    UringOptions o;
    o.queue_depth = number<unsigned>("queue_depth", queue_depth, 0, 32768);
    if (mode == "default")
      o.mode = UringMode::Default;
    else if (mode == "iopoll")
      o.mode = UringMode::IoPoll;
    else if (mode == "sqpoll")
      o.mode = UringMode::SqPoll;
    else if (mode == "sqpoll_iopoll")
      o.mode = UringMode::SqPollIoPoll;
    else
      invalid("mode", "expected default, iopoll, sqpoll, or sqpoll_iopoll");
    o.fixed_files = fixed_files;
    if (read == "normal")
      o.read_mode = UringReadMode::Normal;
    else if (read == "fixed")
      o.read_mode = UringReadMode::Fixed;
    else if (read == "readv_fixed")
      o.read_mode = UringReadMode::ReadvFixed;
    else
      invalid("read_mode", "expected normal, fixed, or readv_fixed");
    if (wait == "block")
      o.wait_mode = UringWaitMode::Block;
    else if (wait == "spin")
      o.wait_mode = UringWaitMode::Spin;
    else
      invalid("wait_mode", "expected block or spin");
    o.sq_thread_idle_ms =
        number<unsigned>("sq_thread_idle_ms", sq_thread_idle_ms, 0, std::numeric_limits<unsigned>::max());
    o.diagnostics = diagnostics;
    if (cuts == "auto")
      o.read_cuts = UringReadCuts::Auto;
    else if (cuts == "0")
      o.read_cuts = UringReadCuts::Off;
    else if (cuts == "1")
      o.read_cuts = UringReadCuts::On;
    else
      invalid("read_cuts", "expected auto, 0, or 1");
    return o;
  }

  // The name an error gives a member, as the group env spells it.
  static std::string key(const char* member) {
    return std::string("SGLANG_EXPERT_STREAM_URING ") + member;
  }

 private:
  [[noreturn]] static void invalid(const char* member, const char* reason) {
    throw std::invalid_argument(key(member) + ": " + reason);
  }
  template <typename T>
  static T number(const char* member, int64_t value, T low, T high) {
    if (value < static_cast<int64_t>(low) || value > static_cast<int64_t>(high))
      invalid(member, "invalid integer or value outside supported range");
    return static_cast<T>(value);
  }
};

// Internal linkage: each module's translation unit owns its installed options, never merged with another library's.
namespace {
struct InstalledUringOptions {
  std::mutex mutex;
  UringOptions options;
};
inline InstalledUringOptions& installed_uring_options_state() {
  static InstalledUringOptions state;
  return state;
}

inline void set_uring_options(const UringOptions& options) {
  auto& state = installed_uring_options_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.options = options;
}

inline UringOptions installed_uring_options() {
  auto& state = installed_uring_options_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.options;
}
}  // namespace

}  // namespace sglang::expert_stream
