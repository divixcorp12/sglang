// One completion mailbox per device-side execution chain. Not part of the service wire ABI.
#pragma once

#include <cstdint>

namespace sglang::expert_stream::wire {

constexpr int64_t kWaitCompletionBytes = 32;
constexpr int64_t kWaitCompletionToken = 0;      // u64: terminal tag in high byte, generation in low 56 bits
constexpr int64_t kWaitCompletionReady = 8;      // u32: stream wait observes 1
constexpr int64_t kWaitCompletionTimeoutNs = 16; // u64 duration, published before the pending token
constexpr uint64_t kWaitTagPending = 0;
constexpr uint64_t kWaitTagReady = 1;
constexpr uint64_t kWaitTagTimeout = 2;
constexpr uint64_t kWaitTagAborted = 3;
constexpr uint64_t kWaitTagBypass = 4;

}  // namespace sglang::expert_stream::wire
