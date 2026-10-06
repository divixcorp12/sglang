// The host half of the lease channel (lease_channel_layout.h). Bodies moved from tier_protocol.h (the word accessors,
// read_record's seqlock copy) and ram_tier.h (copy_completed, cas_gate, open_closed_gate).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#include "../lease_channel_layout.h"

namespace sglang::expert_stream::channel {

// Acquire load of the 32-bit word at `address` in a block shared with the device.
inline uint32_t load_acquire(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
}

// Release store of a 64-bit word shared with the device: a done word, a map-chain tag.
inline void store_release64(uint8_t* address, uint64_t value) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(address), value, __ATOMIC_RELEASE);
}

// Maps sequence 0 to 1. The device never posts sequence 0 (advance wraps 0xFFFFFFFF to 1), so a reader that reached 0
// would spend an iteration on a record nobody posted and store a done word of 0.
inline uint32_t skip_zero(uint32_t seq) {
  return seq == 0 ? 1u : seq;
}

// True when `observed` has reached `seq`, correct across the 2^32 wrap.
inline bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

template <class S>
int64_t ring_index(uint32_t seq) {
  return static_cast<int64_t>((seq - 1u) % S::kRecords);
}

template <class S>
const uint8_t* record_at(const uint8_t* page, uint32_t seq) {
  return page + S::kRing + ring_index<S>(seq) * S::kRecordBytes;
}

// The head word: the last seq the device posted.
template <class S>
uint32_t head(const uint8_t* page) {
  return load_acquire(page + S::kHead);
}

// Copies the record into `raw` if its seq word (the record's first) reads `expected` both before and after the copy:
// false when the writer rewrote it meanwhile. A seqlock: the writer stores the payload, fences, then the seq word last,
// so a record whose seq reads `expected` on both sides is whole. The payload is copied once, at a constant size,
// between the two seq loads: every cache line's load issues together, and nothing after the second seq load reads
// the shared record.
template <class S>
bool read_seqlocked(const uint8_t* record, uint32_t expected, uint8_t (&raw)[S::kRecordBytes]) {
  if (load_acquire(record) != expected) return false;
  std::memcpy(raw, record, S::kRecordBytes);
  std::atomic_thread_fence(std::memory_order_acquire);
  asm volatile("" ::: "memory");  // the copy's plain loads must stay before the seq re-check
  return load_acquire(record) == expected;
}

// The gate word, whose closed bit is set while a wait holds the device stream.
template <class S>
uint32_t gate(const uint8_t* lease) {
  return load_acquire(lease + S::kGate);
}

// Replaces the gate word if it still equals `expected`. A locked cmpxchg on the host line is atomic against the
// device's posted stores to it.
template <class S>
void cas_gate(uint8_t* lease, uint32_t expected, uint32_t desired) {
  __atomic_compare_exchange_n(
      reinterpret_cast<uint32_t*>(lease + S::kGate), &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
}

// Publishes done[G], then opens the gate if the device closed it for G. Dekker with the device's close_gate (gate
// close, fence.sc.sys, done load): done is stored before the gate load, so if the device missed this store it closed
// the gate before this load, which then sees closed(G) and opens it. The host opens only by a CAS from G's own closed
// word, so a stale open for G meets closed(G + k) and changes nothing.
template <class S>
void complete(uint8_t* lease, uint32_t seq, uint64_t gen) {
  store_release64(lease + S::kDone + ring_index<S>(seq) * S::kDoneBytes, gen);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const uint32_t closed = gate_word(seq, kGateClosed);
  if (gate<S>(lease) == closed) cas_gate<S>(lease, closed, gate_word(seq, kGateOpen));
}

// Teardown: with no completer left, a closed wait would hold its stream forever. Opening the gate lets the device
// continue; its commit traps unless done[G] is there, which is acceptable because the process is ending either way.
template <class S>
void open_closed_gate(uint8_t* lease) {
  const uint32_t g = gate<S>(lease);
  if ((g & 0x80000000u) != 0) cas_gate<S>(lease, g, g & ~0x80000000u);
}

}  // namespace sglang::expert_stream::channel
