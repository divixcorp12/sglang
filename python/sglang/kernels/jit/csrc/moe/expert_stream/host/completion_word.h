// The copy engine's completion word: how the copy thread learns that the stream finished a job's copies.
//
// Nothing here calls a driver, takes a lock, allocates or reads a clock, so a CPU test drives every rule below.
//
// See analysis/dsv41-drive/hotpath/results.md section 9c.
#pragma once

#include <cstdint>

namespace sglang::expert_stream {

// The 32-bit completion word the copy stream writes after each job's copies, and the copy thread's poll of it.
//
// The copy stream writes each job's sequence number (its token) into one host-mapped pinned word after the job's
// copies (CudaCopyBackend::mark: cuStreamWriteValue32_v2 with the default flags, which execute the write only after
// the stream's prior work and fence that work's memory before it). The stream runs in order, so a word that has
// reached a job's token means that job and every earlier one completed. The copy thread polls with one acquire load
// of the word: no driver call, and so no libcuda mutex, per poll.
//
// Liveness has no clock: the host has no in-flight timeout, and the watchdog's copy-wait deadline is the timeout. A
// failed copy or write never writes the word, so after `check_every` consecutive pending polls the head asks the
// stream once (`stream_query`: 0 idle, kNotReady busy, anything else a sticky error):
//   - busy: keep polling;
//   - an error: return it, and the engine fails stop;
//   - idle: re-load the word (acquire) and fail stop as a lost write only if it is still short. The first load came
//     before the query, so a write landing between the two would otherwise make a healthy, now idle stream look like
//     a lost write. The stream reports idle only once its work, the word's write included, has completed, and the
//     driver learns that from a completion the GPU writes after the word. Both are posted writes from one engine,
//     so the re-load sees the word.
//
// A poll turn takes ~84 ns in production, so the default budget is ~5.5 ms of one head pending: at most 0.4 stream
// queries per ~2.2 ms copy job.
class CompletionWord {
 public:
  static constexpr int kDone = 0;           // CopyBackend::kDone
  static constexpr int kPending = 1;        // CopyBackend::kPending
  static constexpr int kNotReady = 600;     // CUDA_ERROR_NOT_READY
  static constexpr int kLostWrite = -1001;  // the stream went idle and the word never reached the token
  static constexpr uint32_t kCheckEvery = 1u << 16;

  explicit CompletionWord(uint32_t check_every = kCheckEvery) : check_every_(check_every < 1 ? 1 : check_every) {}

  // Points the poll at the host-mapped word the stream writes.
  void bind(const uint32_t* word) {
    word_ = word;
  }

  // Correct across the 2^32 wrap while fewer than 2^31 jobs are outstanding (at most kCopyRing are).
  static bool reached(uint32_t word, uint32_t token) {
    return static_cast<int32_t>(word - token) >= 0;
  }

  // The word's current value, with acquire ordering.
  uint32_t load() const {
    return __atomic_load_n(word_, __ATOMIC_ACQUIRE);
  }

  // One poll of the head job's `token`. Returns kDone, kPending, kLostWrite, or the stream's error code. Calls
  // `stream_query` at most once per `check_every` consecutive pending polls.
  template <class StreamQuery>
  int poll(uint32_t token, StreamQuery&& stream_query) {
    if (reached(load(), token)) {
      pending_ = 0;
      return kDone;
    }
    if (++pending_ < check_every_) return kPending;
    pending_ = 0;
    const int state = stream_query();
    if (state == kNotReady) return kPending;
    if (state != 0) return state;
    return reached(load(), token) ? kDone : kLostWrite;
  }

 private:
  const uint32_t* word_ = nullptr;
  uint32_t check_every_;
  uint32_t pending_ = 0;  // consecutive pending polls since the last completion or stream query
};

}  // namespace sglang::expert_stream
