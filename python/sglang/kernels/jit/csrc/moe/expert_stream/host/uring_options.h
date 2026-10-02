// The io_uring options of the expert-stream reader, read from SGLANG_EXPERT_STREAM_URING_* environment variables.
//
// Every variable is optional: unset ones keep the production defaults, and an invalid value throws
// std::invalid_argument naming the variable. UringOptions also derives the behaviors that follow from the
// combination of options (polling, wait strategy, read cuts), so the reader asks it rather than re-deriving them.
//
//   QUEUE_DEPTH, MODE, FIXED_FILES, READ_MODE, WAIT_MODE, SQ_THREAD_IDLE_MS, SQ_THREAD_CPU, DIAGNOSTICS, READ_CUTS
#pragma once

#include <charconv>
#include <cstdlib>
#include <limits>
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

// The parsed options. Build one with from_env(); the const members derive the behavior the options imply.
struct UringOptions {
  unsigned queue_depth = 0;  // 0 keeps RowReader's 16 * parts credit limit
  UringMode mode = UringMode::Default;
  bool fixed_files = false;
  UringReadMode read_mode = UringReadMode::Normal;
  UringWaitMode wait_mode = UringWaitMode::Block;
  unsigned sq_thread_idle_ms = 10000;
  int sq_thread_cpu = -1;
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

  // Reads and validates every SGLANG_EXPERT_STREAM_URING_* variable. Throws std::invalid_argument on a bad value.
  static UringOptions from_env() {
    UringOptions o;
    o.queue_depth = number<unsigned>("QUEUE_DEPTH", 0, 0, 32768);
    const auto mode = value("MODE", "default");
    if (mode == "default")
      o.mode = UringMode::Default;
    else if (mode == "iopoll")
      o.mode = UringMode::IoPoll;
    else if (mode == "sqpoll")
      o.mode = UringMode::SqPoll;
    else if (mode == "sqpoll_iopoll")
      o.mode = UringMode::SqPollIoPoll;
    else
      invalid("MODE", "expected default, iopoll, sqpoll, or sqpoll_iopoll");
    o.fixed_files = boolean("FIXED_FILES", false);
    const auto read = value("READ_MODE", "normal");
    if (read == "normal")
      o.read_mode = UringReadMode::Normal;
    else if (read == "fixed")
      o.read_mode = UringReadMode::Fixed;
    else if (read == "readv_fixed")
      o.read_mode = UringReadMode::ReadvFixed;
    else
      invalid("READ_MODE", "expected normal, fixed, or readv_fixed");
    const auto wait = value("WAIT_MODE", "block");
    if (wait == "block")
      o.wait_mode = UringWaitMode::Block;
    else if (wait == "spin")
      o.wait_mode = UringWaitMode::Spin;
    else
      invalid("WAIT_MODE", "expected block or spin");
    o.sq_thread_idle_ms = number<unsigned>("SQ_THREAD_IDLE_MS", 10000, 0, std::numeric_limits<unsigned>::max());
    o.sq_thread_cpu = number<int>("SQ_THREAD_CPU", -1, -1, std::numeric_limits<int>::max());
    if (o.sq_thread_cpu != -1 && !o.sqpoll()) invalid("SQ_THREAD_CPU", "requires a sqpoll mode");
    o.diagnostics = boolean("DIAGNOSTICS", false);
    const auto cuts = value("READ_CUTS", "auto");
    if (cuts == "auto")
      o.read_cuts = UringReadCuts::Auto;
    else if (cuts == "0")
      o.read_cuts = UringReadCuts::Off;
    else if (cuts == "1")
      o.read_cuts = UringReadCuts::On;
    else
      invalid("READ_CUTS", "expected auto, 0, or 1");
    return o;
  }

 private:
  static std::string key(const char* name) {
    return std::string("SGLANG_EXPERT_STREAM_URING_") + name;
  }
  static std::string_view value(const char* name, const char* fallback) {
    const char* found = std::getenv(key(name).c_str());
    return found ? found : fallback;
  }
  [[noreturn]] static void invalid(const char* name, const char* reason) {
    throw std::invalid_argument(key(name) + ": " + reason);
  }
  static bool boolean(const char* name, bool fallback) {
    const auto v = value(name, fallback ? "1" : "0");
    if (v == "0") return false;
    if (v == "1") return true;
    invalid(name, "expected 0 or 1");
  }
  template <typename T>
  static T number(const char* name, T fallback, T low, T high) {
    const char* v = std::getenv(key(name).c_str());
    if (!v) return fallback;
    const std::string_view s(v);
    T result{};
    const auto parsed = std::from_chars(s.data(), s.data() + s.size(), result);
    if (s.empty() || parsed.ec != std::errc{} || parsed.ptr != s.data() + s.size() || result < low || result > high)
      invalid(name, "invalid integer or value outside supported range");
    return result;
  }
};

}  // namespace sglang::expert_stream
