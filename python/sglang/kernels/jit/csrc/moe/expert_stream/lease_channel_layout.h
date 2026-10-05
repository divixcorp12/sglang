// The lease channel: the device-to-host request handoff every expert-stream client shares (LEASE_PROTOCOL.md, "The
// lease channel"). A client is a ChannelSpec: where its head word, record ring, done words and gate sit. The record
// payload, what the host does with it, and the data areas are the client's own.
//
//   page   device-written, host-read: head (u32, the last posted seq, release-stored), then a ring of `Records`
//          records, each a seqlock (seq word first: 0 while rewritten, the seq stored last with a release)
//   lease  host-written, device-read: done[Records] (u64 G = epoch << 32 | seq, release-stored), and the gate (u32)
//
// The gate is the stream's wake-up: closed(G) while a wait holds the stream, open(G) once done[G] holds. The device
// closes it and re-checks done after a system fence; the host stores done, fences, and opens it if it reads closed(G).
// One side always sees the other's store (a Dekker pair), and both open with the same word.
#pragma once

#include <cstdint>

#ifndef SGL_HD
#if defined(__CUDACC__)
#define SGL_HD __host__ __device__
#else
#define SGL_HD
#endif
#endif

namespace sglang::expert_stream::channel {

constexpr uint32_t kGateClosed = 0x80000001u;  // bit 31 set: a cyclic GEQ wait against kGateOpen blocks
constexpr uint32_t kGateOpen = 1;
constexpr uint32_t kGateSeqShift = 2;
constexpr uint32_t kGateSeqMask = 0x1FFFFFFF;

// The gate word for record `seq`: its sequence number in the high bits, the closed/open state in the low bits. An
// open word is in [1, 2^31), so cuStreamWaitValue32's cyclic GEQ against kGateOpen passes it and blocks a closed one.
SGL_HD constexpr uint32_t gate_word(uint32_t seq, uint32_t low) {
  return ((seq & kGateSeqMask) << kGateSeqShift) | low;
}

template <int64_t Head, int64_t Ring, uint32_t Records, int64_t RecordBytes, int64_t Done, int64_t Gate>
struct ChannelSpec {
  static constexpr int64_t kHead = Head;
  static constexpr int64_t kRing = Ring;
  static constexpr uint32_t kRecords = Records;
  static constexpr int64_t kRecordBytes = RecordBytes;
  static constexpr int64_t kDone = Done;
  static constexpr int64_t kDoneBytes = 8;
  static constexpr int64_t kGate = Gate;
  static_assert(Head % 4 == 0 && Ring % 128 == 0 && RecordBytes % 128 == 0, "records are whole 128-byte line pairs");
  static_assert(Records >= 2, "a ring of at least two records");
  static_assert(Done % 8 == 0 && Gate % 128 == 0, "done words are u64; the gate has a line of its own");
  static_assert(Gate >= Done + Records * kDoneBytes || Gate + 4 <= Done, "the gate is not a done word");
};

}  // namespace sglang::expert_stream::channel
