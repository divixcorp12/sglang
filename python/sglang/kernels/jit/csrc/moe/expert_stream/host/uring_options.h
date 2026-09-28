// Explicit io_uring experiments. Unset variables retain the production defaults.
#pragma once

#include <charconv>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sglang::expert_stream {

enum class UringMode { Default, IoPoll, SqPoll, SqPollIoPoll };
enum class UringReadMode { Normal, Fixed, ReadvFixed };
enum class UringWaitMode { Block, Spin };

struct UringOptions {
  unsigned queue_depth = 0;  // 0 keeps RowReader's 16 * parts credit limit.
  UringMode mode = UringMode::Default;
  bool fixed_files = false;
  UringReadMode read_mode = UringReadMode::Normal;
  UringWaitMode wait_mode = UringWaitMode::Block;
  unsigned sq_thread_idle_ms = 1000;
  int sq_thread_cpu = -1;
  bool diagnostics = false;

  bool sqpoll() const {
    return mode == UringMode::SqPoll || mode == UringMode::SqPollIoPoll;
  }
  bool iopoll() const {
    return mode == UringMode::IoPoll || mode == UringMode::SqPollIoPoll;
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
    o.sq_thread_idle_ms = number<unsigned>("SQ_THREAD_IDLE_MS", 1000, 0, std::numeric_limits<unsigned>::max());
    o.sq_thread_cpu = number<int>("SQ_THREAD_CPU", -1, -1, std::numeric_limits<int>::max());
    if (o.sq_thread_cpu != -1 && !o.sqpoll()) invalid("SQ_THREAD_CPU", "requires a sqpoll mode");
    o.diagnostics = boolean("DIAGNOSTICS", false);
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
