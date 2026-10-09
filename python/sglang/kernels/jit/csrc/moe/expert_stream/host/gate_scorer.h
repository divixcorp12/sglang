// The RAM prefetch's gate scorer: the next streamed layer's router on a record's staged input, ranked as
// analysis/dsv41-drive/prefetch-replay/verify_replay.py ranks it (gate_choice and issue's budget).
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::expert_stream {

// IEEE half to float, exact. In software: the host may be built without F16C (SGLANG_JIT_HOST_MARCH).
inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | exponent << 23 | (mantissa & 0x3FFu) << 13;
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | mantissa << 13;
  } else {
    bits = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
  }
  return std::bit_cast<float>(bits);
}

inline float bf16_to_float(uint16_t b) {
  return std::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

// torch's softplus (threshold 20), through log1p as the DSV4.1 router computes it (sqrtsoftplus_log1p).
inline float softplus(float z) {
  return z > 20.0f ? z : std::log1p(std::exp(z));
}

class GateScorer {
 public:
  static constexpr int kDepth = 12;       // each token's ranks walked: the replay's (verify_gate_rankings --depth)
  static constexpr int kMaxPerLayer = 8;  // Python mirror: ram_prefetch.MAX_PER_LAYER

  // Sizes the scratch once, so choose() allocates nothing.
  void reserve(int64_t tokens, int64_t hidden, int64_t experts) {
    tokens_ = tokens;
    hidden_ = hidden;
    experts_ = experts;
    x_.assign(static_cast<size_t>(tokens * hidden), 0.0f);
    score_.assign(static_cast<size_t>(tokens * experts), 0.0f);
    best_.assign(static_cast<size_t>(experts), 0.0f);
    order_.assign(static_cast<size_t>(experts), 0);
    picks_.reserve(static_cast<size_t>(experts));
  }

  // Scores fp16 rows `x_token_bytes` apart against bf16 `w` [experts, hidden] plus `bias`, passing over `skip[e] != 0`;
  // writes up to per_layer experts to `out`, best margin first, and returns how many. Call check_gate_choice first.
  // A NaN score (a NaN input or bias) ranks below every other, in id order; a margin between equal scores is 0.
  // Throws std::invalid_argument for sizes beyond what reserve() sized.
  int choose(
      const uint8_t* x,
      int64_t tokens,
      int64_t x_token_bytes,
      const uint16_t* w,
      const float* bias,
      int64_t experts,
      int64_t hidden,
      int top_k,
      int per_token,
      int per_layer,
      const uint8_t* skip,
      int32_t* out) {
    if (tokens < 0 || tokens > tokens_ || hidden != hidden_ || experts != experts_)
      throw std::invalid_argument("the scorer was reserved for other sizes");
    for (int64_t t = 0; t < tokens; ++t) {
      const auto* row = reinterpret_cast<const uint16_t*>(x + t * x_token_bytes);
      for (int64_t h = 0; h < hidden; ++h)
        x_[t * hidden + h] = half_to_float(row[h]);
    }
    for (int64_t e = 0; e < experts; ++e) {
      const uint16_t* we = w + e * hidden;
      for (int64_t t = 0; t < tokens; ++t) {
        const float score = std::sqrt(softplus(dot(we, &x_[t * hidden], hidden))) + bias[e];
        score_[t * experts + e] = std::isnan(score) ? -std::numeric_limits<float>::infinity() : score;
      }
    }
    std::fill(best_.begin(), best_.begin() + experts, -std::numeric_limits<float>::infinity());
    const int64_t depth = std::min<int64_t>(kDepth, experts);
    for (int64_t t = 0; t < tokens; ++t) {
      const float* s = &score_[t * experts];
      std::iota(order_.begin(), order_.begin() + experts, 0);
      std::partial_sort(order_.begin(), order_.begin() + depth, order_.begin() + experts, [s](int32_t a, int32_t b) {
        return s[a] > s[b] || (s[a] == s[b] && a < b);
      });
      const float kth = s[order_[top_k - 1]];
      int picked = 0;
      for (int64_t i = 0; i < depth && picked < per_token; ++i) {
        const int32_t e = order_[i];
        if (skip[e]) continue;
        // Equal scores (also two infinities) give 0, and -inf - finite stays above best_'s unpicked sentinel.
        const float margin = s[e] == kth ? 0.0f : std::max(s[e] - kth, std::numeric_limits<float>::lowest());
        best_[e] = std::max(best_[e], margin);
        ++picked;
      }
    }
    picks_.clear();
    for (int64_t e = 0; e < experts; ++e)
      if (best_[e] != -std::numeric_limits<float>::infinity()) picks_.push_back(static_cast<int32_t>(e));
    std::sort(picks_.begin(), picks_.end(), [this](int32_t a, int32_t b) {
      return best_[a] > best_[b] || (best_[a] == best_[b] && a < b);
    });
    const int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(per_layer), picks_.size()));
    std::copy_n(picks_.begin(), n, out);
    return n;
  }

 private:
  // Sixteen partial sums in a fixed order: the compiler vectorizes the inner loop, and the result never depends on it.
  static float dot(const uint16_t* w, const float* x, int64_t n) {
    float acc[16] = {};
    int64_t h = 0;
    for (; h + 16 <= n; h += 16)
      for (int j = 0; j < 16; ++j)
        acc[j] += bf16_to_float(w[h + j]) * x[h + j];
    float sum = 0.0f;
    for (int j = 0; j < 16; ++j)
      sum += acc[j];
    for (; h < n; ++h)
      sum += bf16_to_float(w[h]) * x[h];
    return sum;
  }

  int64_t tokens_ = 0, hidden_ = 0, experts_ = 0;  // reserve()'s sizes
  std::vector<float> x_;      // [tokens, hidden]
  std::vector<float> score_;  // [tokens, experts]
  std::vector<float> best_;   // [experts]: an expert's best margin over the tokens, -inf when not picked
  std::vector<int32_t> order_;
  std::vector<int32_t> picks_;
};

// Throws std::invalid_argument for a choice the scorer cannot make.
inline void check_gate_choice(int64_t experts, int64_t top_k, int64_t per_token, int64_t per_layer) {
  if (experts < 1 || experts > 0xFFFF) throw std::invalid_argument("the gate has 1..65535 experts");
  const int64_t depth = std::min<int64_t>(GateScorer::kDepth, experts);
  if (top_k < 1 || top_k > depth) throw std::invalid_argument("top_k must be in 1.." + std::to_string(depth));
  if (per_token < 1 || per_token > GateScorer::kDepth)
    throw std::invalid_argument("per_token must be in 1.." + std::to_string(GateScorer::kDepth));
  if (per_layer < 1 || per_layer > GateScorer::kMaxPerLayer)
    throw std::invalid_argument("per_layer must be in 1.." + std::to_string(GateScorer::kMaxPerLayer));
}

}  // namespace sglang::expert_stream
