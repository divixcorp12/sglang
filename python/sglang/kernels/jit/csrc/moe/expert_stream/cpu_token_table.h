// The token table of a CPU experts input row that holds several tokens (plan
// 2026-10-06-dsv41-dspark-both-cpu-experts, a DSpark verify).
//
// A row holds tokens_max staged inputs, x_token_bytes apart (fp16, each padded to 16 bytes). With tokens_max > 1 the
// table follows them; the post kernel writes it for every lane of the record:
//   +0                                       u32  the record's tokens, 1..tokens_max
//   +kHeaderBytes + 4 j                      u32  lane j's token mask: bit t when token t routes lane j's expert
//   +kHeaderBytes + 4 lanes + 4 (t lanes + j)  u32  the fp32 bits of token t's routing weight for lane j's expert
// The CPU expert thread reads it to run one forward of `tokens` rows, each with its own weights
// (CpuExpertEngine::run_job). A one-token row has no table: the record's lane weight serves. The Python mirror is
// expert_lease_block.cpu_row_bytes.
#pragma once

#include <cstdint>

namespace sglang::expert_stream {

struct CpuTokenTable {
  static constexpr int64_t kMaxTokens = 32;    // one u32 mask bit per token
  static constexpr int64_t kHeaderBytes = 16;  // the token count, padded
};

}  // namespace sglang::expert_stream
