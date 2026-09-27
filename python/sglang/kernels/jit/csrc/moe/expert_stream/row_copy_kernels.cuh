// Row-copy kernels: piece-streaming helpers and the stream kernel, and the copy wait (exl3_ram_miss.cuh split).
#pragma once

#include "lease_device.cuh"
#include "row_layout.h"
#include "tensor_checks.h"

#include <sgl_kernel/tensor.h>

#include <bit>

namespace sglang {

namespace device::expert_stream {

// Piece streaming's stream kernel S (piece-streaming plan section 5): kStreamBlocks blocks of kStreamThreads.
constexpr int kStreamBlocks = 8;
constexpr int kStreamThreads = 256;
constexpr int kRowPieces = 8;  // the host's kPieces: one readiness bit per piece of a row
constexpr uint32_t kAllPieces = 255u;
// S's one counter word: finished blocks in the low 16 bits, completed blocks in units of kStreamCompleted above them.
// One word, so the last block reads both counts in the atomic that makes it last.
constexpr uint32_t kStreamCompleted = 65536u;
constexpr uint32_t kStreamFinishedMask = 65535u;
// Test-only fault words (a device int32 tensor, all zero in production).
constexpr int kStreamFaultAbortBlock = 0;  // 1 + the block that takes the abort path (0: none)
constexpr int kStreamFaultAbortDelay = 1;  // ns that block spins before its abort stores
constexpr int kStreamFaultStall = 2;       // ns every leader pass stalls between reading the masks and kDemandDone
constexpr int kStreamFaultCountDelay = 3;  // ns a completing block spins before its count
constexpr int kStreamFaultWords = 4;

// state[*word] += value, clamped at INT32_MAX like W2's kPolls, from any number of blocks at once.
SGL_DEVICE void saturating_add(int32_t* word, int64_t value) {
  int32_t old = *reinterpret_cast<volatile int32_t*>(word);
  while (true) {
    const int64_t sum = static_cast<int64_t>(old) + value;
    const int32_t next = static_cast<int32_t>(sum < 0x7fffffffLL ? sum : 0x7fffffffLL);
    const int32_t seen = atomicCAS(word, old, next);
    if (seen == old) return;
    old = seen;
  }
}

SGL_DEVICE void spin_ns(int64_t ns) {
  if (ns <= 0) return;
  const uint64_t until = global_ns() + static_cast<uint64_t>(ns);
  while (static_cast<int64_t>(global_ns() - until) < 0) __nanosleep(256);
}

// ld.global.cv, never .nc: a tag-2 lane's host bytes are written while the kernel runs, and .nc may serve a line
// cached before its piece was published (LEASE_PROTOCOL.md E1 amendment).
SGL_DEVICE void stream_copy16(const uint8_t* src, uint8_t* dst) {
  uint64_t lo, hi;
  asm volatile("ld.global.cv.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
  asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(dst), "l"(lo), "l"(hi) : "memory");
}

SGL_DEVICE void stream_copy1(const uint8_t* src, uint8_t* dst) {
  uint16_t value;
  asm volatile("ld.global.cv.u8 %0, [%1];" : "=h"(value) : "l"(src) : "memory");
  *dst = static_cast<uint8_t>(value);
}

// This block's share of `bytes` bytes: units (16 B when both ends allow it) in chunks of kStreamThreads, the chunks
// dealt round-robin over the grid, so every block copies about 1/gridDim of a range and no two blocks write one byte.
SGL_DEVICE void stream_copy_slice(const uint8_t* src, uint8_t* dst, int64_t bytes) {
  const int64_t tid = threadIdx.x;
  const bool aligned = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0;
  const int64_t units = aligned ? bytes / 16 : bytes;
  for (int64_t chunk = blockIdx.x; chunk * kStreamThreads < units; chunk += gridDim.x) {
    const int64_t u = chunk * kStreamThreads + tid;
    if (u >= units) break;
    if (aligned) {
      stream_copy16(src + 16 * u, dst + 16 * u);
    } else {
      stream_copy1(src + u, dst + u);
    }
  }
  if (aligned && blockIdx.x == 0) {
    for (int64_t b = units * 16 + tid; b < bytes; b += kStreamThreads) stream_copy1(src + b, dst + b);
  }
}

// This block's slice of piece `piece` of one lane. `runs` is the lane's [kRowPieces][row_segments][2] table of
// name-row byte ranges; `segment_map` gives each row segment's copy-table entry (-1: none), then a flag per entry
// that no row segment names, which is copied whole with piece 0 (C2 copied every entry for a lane).
SGL_DEVICE void stream_copy_piece(
    const int64_t* segments,
    int64_t segment_count,
    const int32_t* segment_map,
    int64_t row_segments,
    const int32_t* runs,
    int piece,
    int64_t host_slot,
    int64_t dst_slot) {
  for (int64_t r = 0; r < row_segments; ++r) {
    const int32_t entry = segment_map[r];
    if (entry < 0) continue;
    const int64_t lo = runs[(piece * row_segments + r) * 2];
    const int64_t hi = runs[(piece * row_segments + r) * 2 + 1];
    if (hi <= lo) continue;
    const int64_t* e = segments + 3 * entry;
    const int64_t row_bytes = e[2];
    stream_copy_slice(
        reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(e[0])) + host_slot * row_bytes + lo,
        reinterpret_cast<uint8_t*>(static_cast<intptr_t>(e[1])) + dst_slot * row_bytes + lo,
        hi - lo);
  }
  if (piece != 0) return;
  for (int64_t k = 0; k < segment_count; ++k) {
    if (segment_map[row_segments + k] == 0) continue;
    const int64_t* e = segments + 3 * k;
    const int64_t row_bytes = e[2];
    stream_copy_slice(
        reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(e[0])) + host_slot * row_bytes,
        reinterpret_cast<uint8_t*>(static_cast<intptr_t>(e[1])) + dst_slot * row_bytes,
        row_bytes);
  }
}

struct StreamLanes {
  int32_t mine[kLeaseLanes];  // planned and not claimed by W1: this kernel's lane
  int32_t admitted[kLeaseLanes];
  int32_t loading[kLeaseLanes];  // admitted under tag LOADING (else READY: every piece is there)
  int32_t slot[kLeaseLanes];
  uint32_t slot_generation[kLeaseLanes];
  uint32_t done[kLeaseLanes];  // pieces this block has copied its slice of
  uint32_t todo[kLeaseLanes];  // pieces to copy this pass
  int identity;                // a published lane failed validation
  int aborting;
  uint32_t reason;             // the abort's kLeaseReason*, 0 for one that names none
  int served;                  // kDemandDone >= seq, status kServed, and the re-read found every mask full
  int finished;                // served and every piece of every lane copied
  int probed;
};

// One lane's admission (plan 5): the whole lane_result_valid contract, tag LOADING accepted as well as READY.
// Returns false for a published lane that fails it; an unpublished lane is simply not admitted yet.
SGL_DEVICE bool stream_admit(
    StreamLanes& sh,
    int lane,
    const uint8_t* results,
    uint64_t generation,
    int64_t expected_expert,
    int64_t experts,
    uint32_t capacity) {
  bool ready_seen = false;
  bool loading = false;
  bool copying = false;
  int32_t slot = 0;
  uint32_t slot_generation = 0;
  if (lane_result_valid(
          results + lane * kLeaseRowResultBytes, generation, expected_expert, capacity, &slot, &slot_generation,
          &ready_seen, /*accept_loading=*/true, &loading, /*accept_copying=*/true, &copying) &&
      expected_expert >= 0 && expected_expert < experts) {
    if (copying) {
      // A copy-engine lane W1 did not claim (its budget ran out first): the copy wait owns it, not this kernel.
      sh.mine[lane] = 0;
      return true;
    }
    sh.slot[lane] = slot;
    sh.slot_generation[lane] = slot_generation;
    sh.loading[lane] = loading ? 1 : 0;
    sh.admitted[lane] = 1;
    return true;
  }
  return !ready_seen;
}

SGL_DEVICE uint32_t piece_bits(uint64_t word, uint64_t generation) {
  return (word >> 8) == (generation & ((1ull << 56) - 1)) ? static_cast<uint32_t>(word & 0xFFu) : 0u;
}

}  // namespace device::expert_stream

// Piece streaming's stage 2 (piece-streaming plan section 5): replaces W2 and C2. It covers every planned lane W1 did
// not claim, admits each once its RowResult validates (tag READY or LOADING), and copies each piece as its bit appears
// in the lane's PieceMask word, so the copy overlaps the read. Every block runs the same leader loop on its own and
// copies its own slice of every piece; there is no inter-block barrier, so no co-residency is assumed.
//
// Commit (plan 5, C1): each block ends on exactly one path and counts into one word. The block that makes the count
// finished commits `go_2` only if every block completed, none aborted, and (by completing) it saw kDemandDone >= seq
// with status kServed and every mask full on a re-read made after acquiring kDemandDone. Otherwise go_2 stays at W1's
// reset of 0. Like W2 it never writes `keep` (the finalize kernel's alone), a terminal or the fatal word.
struct StreamParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  const int32_t* dst_slots;
  int64_t row;
  int64_t lanes;
  int64_t experts;
  int64_t* host_rows_2;
  int32_t* dst_slots_2;
  int64_t* ram_miss;
  uint8_t* lease;
  int64_t lease_d;
  int64_t lease_p;
  const int32_t* claimed;
  int32_t* go_2;
  int64_t* lane_ctx_2;
  int32_t* origin_2;
  uint32_t* stream_count;
  int32_t* stream_abort;
  const int64_t* segments;
  int64_t segment_count;
  const int32_t* segment_map;
  int64_t row_segments;
  const int32_t* piece_runs;
  const int32_t* fault;
};

__global__ __launch_bounds__(device::expert_stream::kStreamThreads, 1) void exl3_ram_miss_lease_stream_kernel(
    const __grid_constant__ StreamParams p) {
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int64_t* __restrict__ const planned = p.planned;
  const int32_t* __restrict__ const count = p.count;
  const int32_t* __restrict__ const dst_slots = p.dst_slots;
  const int64_t row = p.row;
  const int64_t lanes = p.lanes;
  const int64_t experts = p.experts;
  int64_t* __restrict__ const host_rows_2 = p.host_rows_2;
  int32_t* __restrict__ const dst_slots_2 = p.dst_slots_2;
  int64_t* __restrict__ const ram_miss = p.ram_miss;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  const int64_t lease_p = p.lease_p;
  const int32_t* __restrict__ const claimed = p.claimed;
  int32_t* __restrict__ const go_2 = p.go_2;
  int64_t* __restrict__ const lane_ctx_2 = p.lane_ctx_2;
  int32_t* __restrict__ const origin_2 = p.origin_2;
  uint32_t* __restrict__ const stream_count = p.stream_count;
  int32_t* __restrict__ const stream_abort = p.stream_abort;
  const int64_t* __restrict__ const segments = p.segments;
  const int64_t segment_count = p.segment_count;
  const int32_t* __restrict__ const segment_map = p.segment_map;
  const int64_t row_segments = p.row_segments;
  const int32_t* __restrict__ const piece_runs = p.piece_runs;
  const int32_t* __restrict__ const fault = p.fault;
  using namespace device::expert_stream;
  __shared__ StreamLanes sh;
  __shared__ int64_t planned_count;
  __shared__ uint32_t seq;
  __shared__ uint64_t generation;
  __shared__ uint32_t capacity;
  __shared__ int started;         // W2's `ok && seq != 0` at entry: the requests W2 counts in kWaits
  __shared__ int64_t unclaimed;   // planned lanes W1 did not claim: W2's `unclaimed`, whatever path S takes
  const int tid = threadIdx.x;

  if (tid == 0) {
    planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
    seq = static_cast<uint32_t>(state[kPending]);
    generation = seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
    sh.identity = 0;
    sh.aborting = 0;
    sh.reason = 0;
    sh.served = 0;
    sh.finished = 0;
    sh.probed = 0;
    bool ok = state[kSticky] == 0 && state[kReqFailed] == 0 && ld_acquire_sys(page + kFatal) == 0 &&
              ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
    uint32_t reason = ok ? 0u : kLeaseReasonAborted;
    if (ok && (planned_count > kLeaseLanes || planned_count > lanes)) {
      ok = false;
      reason = kLeaseReasonCount;
    }
    if (ok && seq == 0 && planned_count > 0) {
      // Lanes and no armed request to have leased them: a protocol error, as in W2 (it names no reason).
      if (blockIdx.x == 0) state[kFailures] += 1;
      ok = false;
      reason = 0;
    }
    started = ok && seq != 0 ? 1 : 0;
    unclaimed = 0;
    for (int64_t i = 0; i < min(planned_count, static_cast<int64_t>(kLeaseLanes)); ++i) {
      if (claimed[i] == 0) ++unclaimed;
    }
    if (ok && fault[kStreamFaultAbortBlock] == static_cast<int32_t>(blockIdx.x) + 1) {
      ok = false;
      reason = kLeaseReasonAborted;
    }
    if (!ok) {
      sh.aborting = 1;
      sh.reason = reason;
    } else if (seq == 0) {
      sh.served = 1;  // no request and no lanes: nothing to wait for or copy
    }
    capacity = seq != 0 ? *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4)
                        : 0u;
  }
  __syncthreads();
  if (tid < kLeaseLanes) {
    sh.mine[tid] = sh.aborting == 0 && tid < planned_count && claimed[tid] == 0 ? 1 : 0;
    sh.admitted[tid] = 0;
    sh.loading[tid] = 0;
    sh.slot[tid] = 0;
    sh.slot_generation[tid] = 0;
    sh.done[tid] = 0;
    sh.todo[tid] = 0;
  }
  __syncthreads();

  const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
  const uint8_t* results = lease + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
  const uint8_t* masks = lease + lease_p + idx * kLeaseLanes * kLeasePieceMaskLineBytes;
  const uint64_t deadline = load_deadline(state);
  const int64_t piece_stride = static_cast<int64_t>(kRowPieces) * row_segments * 2;
  int64_t polls = 0;
  int64_t pieces = 0;

  while (sh.aborting == 0 && sh.finished == 0) {
    // Admission and masks: one leader-warp lane per request lane. A READY lane (a hit W1 missed) has every piece.
    if (tid < kLeaseLanes && sh.mine[tid] != 0) {
      if (sh.admitted[tid] == 0 && !stream_admit(sh, tid, results, generation, planned[tid], experts, capacity)) {
        sh.identity = 1;
      }
      if (sh.admitted[tid] != 0) {
        const uint32_t bits =
            sh.loading[tid] != 0 ? piece_bits(ld_acquire_sys64(masks + tid * kLeasePieceMaskLineBytes), generation)
                                 : kAllPieces;
        sh.todo[tid] = bits & ~sh.done[tid];
      }
    }
    __syncthreads();
    bool copied = false;
    if (sh.identity == 0) {
      for (int lane = 0; lane < kLeaseLanes; ++lane) {
        const uint32_t todo = sh.todo[lane];
        if (todo == 0) continue;
        copied = true;
        const int32_t* runs = piece_runs + (row * experts + planned[lane]) * piece_stride;
        for (int piece = 0; piece < kRowPieces; ++piece) {
          if ((todo >> piece & 1u) == 0) continue;
          stream_copy_piece(
              segments, segment_count, segment_map, row_segments, runs, piece, sh.slot[lane], dst_slots[lane]);
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      for (int lane = 0; lane < kLeaseLanes; ++lane) {
        pieces += __popc(sh.todo[lane]);
        sh.done[lane] |= sh.todo[lane];
        sh.todo[lane] = 0;
      }
      if (copied && sh.probed == 0) {
        // The host's only view of streaming progress (G2): stored once this block's first slice is copied.
        st_release_sys64(
            lease + lease_d + kLeaseStreamProbe + idx * kLeaseStreamProbeBytes, tagged_word(kLeaseTagStreamed, generation));
        sh.probed = 1;
      }
      ++polls;
      if (sh.identity != 0) {
        sh.aborting = 1;
        sh.reason = kLeaseReasonIdentity;
      } else if (sh.served == 0) {
        spin_ns(fault[kStreamFaultStall]);
        if (static_cast<int64_t>(global_ns() - deadline) >= 0) {
          sh.aborting = 1;
          sh.reason = kLeaseReasonTimeout;
        } else if (ld_acquire_sys(page + kFatal) != 0 || ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0) {
          sh.aborting = 1;
          sh.reason = kLeaseReasonAborted;
        } else if (reached(ld_acquire_sys(page + kDemandDone), seq)) {
          // The acquire orders this thread's status and mask loads below after it: the other threads' copies follow
          // through the block barrier at the end of the pass. No fence needed.
          const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
          const uint16_t status = *reinterpret_cast<const volatile uint16_t*>(record + kRecStatus);
          if (status != kServed) {
            sh.aborting = 1;
            sh.reason = kLeaseReasonFailed;
          } else {
            // The judgement, on masks re-read by this thread AFTER its acquire of kDemandDone: the host stores
            // kDemandDone after read() returned, so after every publish CAS, and a mask read earlier in the pass may
            // predate the last one. After kDemandDone every lane is granted, so an unadmitted one is a violation too.
            bool whole = true;
            for (int lane = 0; lane < kLeaseLanes && whole; ++lane) {
              if (sh.mine[lane] == 0) continue;
              if (sh.admitted[lane] == 0) stream_admit(sh, lane, results, generation, planned[lane], experts, capacity);
              if (sh.mine[lane] == 0) continue;  // admitted as a copy-engine lane just now
              if (sh.admitted[lane] == 0) {
                whole = false;
                break;
              }
              const uint32_t bits = sh.loading[lane] != 0
                                        ? piece_bits(ld_acquire_sys64(masks + lane * kLeasePieceMaskLineBytes), generation)
                                        : kAllPieces;
              if (bits != kAllPieces) whole = false;
            }
            if (whole) {
              sh.served = 1;
            } else {
              sh.aborting = 1;
              sh.reason = kLeaseReasonIdentity;
            }
          }
        } else if (!copied) {
          __nanosleep(256);
        }
      }
      // Once served there is no D5 check: termination rests on the masks staying final until the ring slot's reuse
      // re-initialises them, 16 requests later, which cannot happen while this request is still in the chain.
      if (sh.aborting == 0 && sh.served != 0) {
        bool all = true;
        for (int lane = 0; lane < kLeaseLanes; ++lane) {
          if (sh.mine[lane] != 0 && sh.done[lane] != kAllPieces) all = false;
        }
        if (all) sh.finished = 1;
      }
    }
    __syncthreads();
  }

  if (tid != 0) return;
  saturating_add(&state[kStreamPolls], polls);
  if (blockIdx.x == 0) atomicAdd(&state[kStreamPieces], static_cast<int32_t>(pieces));
  uint32_t increment;
  if (sh.aborting != 0) {
    if (fault[kStreamFaultAbortBlock] == static_cast<int32_t>(blockIdx.x) + 1) spin_ns(fault[kStreamFaultAbortDelay]);
    // The abort path: this block's own failure record, first writer of the reason wins, then abort, fence, count.
    *reinterpret_cast<volatile int32_t*>(&state[kReqFailed]) = 1;
    if (sh.reason != 0 && atomicCAS(&state[kFailReason], 0, static_cast<int32_t>(sh.reason)) == 0) {
      if (sh.reason == kLeaseReasonTimeout) state[kTimeouts] += 1;
      if (sh.reason == kLeaseReasonFailed) state[kFailures] += 1;
      if (sh.reason == kLeaseReasonIdentity) state[kUnservedMisses] += static_cast<int32_t>(unclaimed);
    }
    *reinterpret_cast<volatile int32_t*>(stream_abort) = 1;
    __threadfence();
    increment = 1u;
  } else {
    spin_ns(fault[kStreamFaultCountDelay]);
    __threadfence();  // this block's copies, before its count says it completed
    increment = 1u + kStreamCompleted;
  }
  const uint32_t old = atomicAdd(stream_count, increment);
  if ((old & kStreamFinishedMask) != gridDim.x - 1) return;
  // The last block. It decides from the value its own atomic returned, never from a second read of the counter.
  const uint32_t now = old + increment;
  __threadfence();
  int32_t aborted;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(aborted) : "l"(stream_abort) : "memory");
  if (started != 0) state[kWaits] += 1;
  const bool commit = now / kStreamCompleted == gridDim.x && aborted == 0 && sh.aborting == 0 && sh.served != 0;
  if (!commit) {
    ram_miss[0] += unclaimed;  // nothing was served for the lanes W1 did not claim
    return;
  }
  int64_t n = 0;
  for (int lane = 0; lane < kLeaseLanes; ++lane) {
    if (sh.mine[lane] == 0) continue;
    host_rows_2[n] = static_cast<int64_t>(sh.slot[lane]);
    dst_slots_2[n] = dst_slots[lane];
    origin_2[n] = static_cast<int32_t>(lane);
    lane_ctx_2[4 * n + 0] = static_cast<int64_t>(generation);
    lane_ctx_2[4 * n + 1] = static_cast<int64_t>(sh.slot_generation[lane]);
    lane_ctx_2[4 * n + 2] = row;
    lane_ctx_2[4 * n + 3] = static_cast<int64_t>(sh.slot[lane]);
    ++n;
  }
  for (int64_t i = n; i < lanes; ++i) host_rows_2[i] = 0;
  go_2[0] = static_cast<int32_t>(n);  // the single commit point
}


// The copy wait's SM reads: `bytes` of the pinned slot into the destination, four 16-byte units in flight per thread
// (ld.global.cv, as S: never .nc on host bytes). Every load has returned once its store is issued.
SGL_DEVICE void copy_wait_read(const uint8_t* src, uint8_t* dst, int64_t bytes) {
  const bool aligned = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0;
  const int64_t units = aligned ? bytes / 16 : 0;
  const int64_t step = blockDim.x;
  int64_t u = threadIdx.x;
  for (; u + 3 * step < units; u += 4 * step) {
    uint64_t v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      asm volatile("ld.global.cv.v2.b64 {%0,%1},[%2];"
                   : "=l"(v[2 * k]), "=l"(v[2 * k + 1])
                   : "l"(src + 16 * (u + k * step))
                   : "memory");
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(dst + 16 * (u + k * step)), "l"(v[2 * k]), "l"(v[2 * k + 1])
                   : "memory");
    }
  }
  for (; u < units; u += step) device::expert_stream::stream_copy16(src + 16 * u, dst + 16 * u);
  for (int64_t b = units * 16 + threadIdx.x; b < bytes; b += step) device::expert_stream::stream_copy1(src + b, dst + b);
}

// Copy-engine wait (LEASE_PROTOCOL.md 7.6), after S and A2 and before F. The COPYING lanes are read back from the row
// results rather than from W1's claims, because S also hands over the ones W1's budget missed. It commits go_ce only
// once CopyDone carries this generation and exactly that lane mask; any other exit leaves go_ce 0 and records the
// failure for F, which publishes the terminal. It never writes keep, a terminal or the fatal word.
//
// SM small copies (`sm_count` > 0, SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES): the copy engine copied only the
// row's other entries, so first the whole block reads the `sm_count` entries of `sm_table` ({source slab, destination
// tensor, row bytes}) of every COPYING lane from its leased host slot into its destination slot, then thread 0 fences
// and publishes SmAck. The service releases those leases only after SmAck, so no slot is rewritten under these
// reads. SmAck is published for every armed request, also one that failed and read nothing, since a lease it holds
// is released only by it; nothing of this request is read after it.
struct CopyWaitParams {
  uint8_t* page;
  int32_t* state;
  const int32_t* count;
  uint8_t* lease;
  int64_t lease_c;
  int64_t lease_d;
  const int64_t* sm_table;
  int64_t sm_count;
  int32_t* go_ce;
};

__global__ __launch_bounds__(device::expert_stream::kCopyWaitThreads, 1) void exl3_ram_miss_lease_copy_wait_kernel(
    const __grid_constant__ CopyWaitParams p) {
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int32_t* __restrict__ const count = p.count;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_c = p.lease_c;
  const int64_t lease_d = p.lease_d;
  const int64_t* __restrict__ const sm_table = p.sm_table;
  const int64_t sm_count = p.sm_count;
  int32_t* __restrict__ const go_ce = p.go_ce;
  using namespace device::expert_stream;
  __shared__ uint32_t sm_mask;  // the lanes whose SM entries this kernel read; the commit must name exactly them
  if (sm_count > 0) {
    __shared__ int32_t sm_host[kLeaseLanes];
    __shared__ int32_t sm_dst[kLeaseLanes];
    const uint32_t seq = static_cast<uint32_t>(state[kPending]);
    const uint64_t generation = (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq;
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    if (threadIdx.x == 0) {
      uint32_t mask = 0;
      const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
      if (seq != 0 && planned_count != 0 && state[kReqFailed] == 0 && state[kSticky] == 0) {
        const uint8_t* results = lease + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
        const uint8_t* request = lease + lease_d + kLeaseLaneRequest + idx * kLeaseLaneRequestBytes;
        const int64_t named = planned_count < kLeaseLanes ? planned_count : kLeaseLanes;
        const uint64_t generation_mask = (1ull << 56) - 1;
        for (int64_t lane = 0; lane < named; ++lane) {
#ifdef EXL3_RAM_MISS_TEST_CW_SM_SKIP_LANE
          // Test build: this lane reads as not yet COPYING here, as if it turned COPYING after the SM read.
          if (lane == EXL3_RAM_MISS_TEST_CW_SM_SKIP_LANE) continue;
#endif
          const uint8_t* result = results + lane * kLeaseRowResultBytes;
          const uint64_t word = ld_acquire_sys64(result + kLeaseRrReady);
          if ((word >> 56) != kLeaseTagCopying || (word & generation_mask) != generation) continue;
          // After the acquire of the ready word; a COPYING lane's payload is fixed until its lease is released.
          sm_host[lane] = *reinterpret_cast<const volatile int32_t*>(result + kLeaseRrHostSlot);
          sm_dst[lane] = *reinterpret_cast<const volatile int32_t*>(request + kLeaseLrDst + 4 * lane);
          mask |= 1u << lane;
        }
      }
      sm_mask = mask;
    }
    __syncthreads();
#ifdef EXL3_RAM_MISS_TEST_CW_SM_READ_DELAY_NS
    // Test build: all but the first warp start their reads late, so an SmAck that does not wait for them shows.
    if (threadIdx.x >= 32) spin_ns(EXL3_RAM_MISS_TEST_CW_SM_READ_DELAY_NS);
#endif
    for (uint32_t lanes = sm_mask; lanes != 0; lanes &= lanes - 1) {
      const int lane = __ffs(lanes) - 1;
      for (int64_t k = 0; k < sm_count; ++k) {
        const int64_t* e = sm_table + 3 * k;
        copy_wait_read(
            reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(e[0])) + sm_host[lane] * e[2],
            reinterpret_cast<uint8_t*>(static_cast<intptr_t>(e[1])) + sm_dst[lane] * e[2],
            e[2]);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && seq != 0) {
      __threadfence_system();
      st_release_sys64(lease + lease_d + kLeaseSmAck + idx * kLeaseSmAckBytes, tagged_word(kLeaseTagSmAck, generation));
    }
  }
  if (threadIdx.x != 0) return;
  go_ce[0] = 0;  // fail closed
  const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  // An earlier stage's failure is F's to publish; a request that never armed has no copy-engine lanes.
  if (seq == 0 || planned_count == 0 || state[kReqFailed] != 0 || state[kSticky] != 0) return;
  const uint64_t generation = (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq;
  const uint64_t generation_mask = (1ull << 56) - 1;
  const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
  const uint8_t* results = lease + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
  const int64_t named = planned_count < kLeaseLanes ? planned_count : kLeaseLanes;
  uint32_t mask = 0;
  for (int64_t lane = 0; lane < named; ++lane) {
    const uint64_t word = ld_acquire_sys64(results + lane * kLeaseRowResultBytes + kLeaseRrReady);
    if ((word >> 56) == kLeaseTagCopying && (word & generation_mask) == generation) mask |= 1u << lane;
  }
  if (sm_count > 0 && mask != sm_mask) {
    // A lane COPYING now but not at the SM read never had its SM entries read: committing it would pair the DMA's
    // fresh tensors with stale small ones. Fail closed; SmAck is already published, so its lease still retires.
    state[kReqFailed] = 1;
    if (state[kFailReason] == 0) state[kFailReason] = static_cast<int32_t>(kLeaseReasonIdentity);
    return;
  }
  if (mask == 0) return;
  state[kCopyWaits] += 1;
  const uint8_t* done = lease + lease_c + idx * kLeaseCopyDoneBytes;
  const uint64_t expected = tagged_word(kLeaseTagCopied, generation);
  const uint64_t deadline = load_deadline(state);
  uint32_t reason = 0;
  uint64_t word = ld_acquire_sys64(done + kLeaseCdGen);
  if (word != expected) state[kCopySpun] += 1;
  while (word != expected) {
    if (static_cast<int64_t>(global_ns() - deadline) >= 0) {
      reason = kLeaseReasonTimeout;
      break;
    }
    if (ld_acquire_sys(page + kFatal) != 0 || ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0) {
      reason = kLeaseReasonAborted;
      break;
    }
    __nanosleep(256);
    word = ld_acquire_sys64(done + kLeaseCdGen);
  }
  // The acquire of the tagged word orders this load after it: the service stores the mask first.
  if (reason == 0 && *reinterpret_cast<const volatile uint32_t*>(done + kLeaseCdMask) != mask) {
    reason = kLeaseReasonIdentity;
  }
  if (reason != 0) {
    state[kReqFailed] = 1;
    if (state[kFailReason] == 0) state[kFailReason] = static_cast<int32_t>(reason);
    if (reason == kLeaseReasonTimeout) state[kTimeouts] += 1;
    return;
  }
  go_ce[0] = __popc(mask);  // the single commit point
}

// Checked host launchers for the row-copy kernels above (mechanical-refactor-verify Task 8), templated on the
// streamed row's compile-time layout facts (name count, small-tensor mask). FFI signatures are unchanged from the
// free launchers they replace.
template <expert_stream::ExpertRowLayout L>
struct RowCopyKernel {
  static void lease_stream(
      tvm::ffi::TensorView page,
      tvm::ffi::TensorView state,
      tvm::ffi::TensorView planned,
      tvm::ffi::TensorView count,
      tvm::ffi::TensorView dst_slots,
      int64_t row,
      int64_t experts,
      tvm::ffi::TensorView host_rows_2,
      tvm::ffi::TensorView dst_slots_2,
      tvm::ffi::TensorView ram_miss,
      int64_t lease_address,
      int64_t lease_d,
      int64_t lease_p,
      tvm::ffi::TensorView claimed,
      tvm::ffi::TensorView go_2,
      tvm::ffi::TensorView lane_ctx_2,
      tvm::ffi::TensorView origin_2,
      tvm::ffi::TensorView stream_count,
      tvm::ffi::TensorView stream_abort,
      tvm::ffi::TensorView segments,
      tvm::ffi::TensorView segment_map,
      int64_t row_segments,
      tvm::ffi::TensorView piece_runs,
      tvm::ffi::TensorView fault) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto P_ = SymbolicSize{"planned"};
    const int64_t lanes = std::min<int64_t>(host_rows_2.size(0), dst_slots.size(0));

    expert_stream::verify_named("page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device(on_host), page);
    expert_stream::verify_named(
        "state", TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device(device), state);
    expert_stream::verify_named("planned", TensorMatcher({P_}).with_dtype<int64_t>().with_device(device), planned);
    RuntimeCheck(P_.unwrap() >= lanes, "planned: must have at least as many lanes as host_rows_2/dst_slots");
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device(device), count);
    expert_stream::verify_named("dst_slots", TensorMatcher({-1}).with_dtype<int32_t>().with_device(device), dst_slots);
    expert_stream::verify_named(
        "host_rows_2", TensorMatcher({-1}).with_dtype<int64_t>().with_device(device), host_rows_2);
    expert_stream::verify_named(
        "dst_slots_2", TensorMatcher({-1}).with_dtype<int32_t>().with_device(device), dst_slots_2);
    expert_stream::verify_named("ram_miss", TensorMatcher({-1}).with_dtype<int64_t>().with_device(device), ram_miss);
    expert_stream::verify_named("claimed", TensorMatcher({-1}).with_dtype<int32_t>().with_device(device), claimed);
    expert_stream::verify_named("go_2", TensorMatcher({1}).with_dtype<int32_t>().with_device(device), go_2);
    expert_stream::verify_named(
        "lane_ctx_2", TensorMatcher({-1, 4}).with_dtype<int64_t>().with_device(device), lane_ctx_2);
    expert_stream::verify_named("origin_2", TensorMatcher({-1}).with_dtype<int32_t>().with_device(device), origin_2);
    expert_stream::verify_named(
        "stream_count", TensorMatcher({1}).with_dtype<int32_t>().with_device(device), stream_count);
    expert_stream::verify_named(
        "stream_abort", TensorMatcher({1}).with_dtype<int32_t>().with_device(device), stream_abort);
    expert_stream::verify_named(
        "segments",
        TensorMatcher({expert_stream::kNumNames<L>, 3}).with_dtype<int64_t>().with_device(device),
        segments);
    expert_stream::verify_named(
        "segment_map", TensorMatcher({-1}).with_dtype<int32_t>().with_device(device), segment_map);
    RuntimeCheck(
        segment_map.size(0) == row_segments + expert_stream::kNumNames<L>,
        "segment_map: size must equal row_segments + the layout's name count");
    expert_stream::verify_named(
        "piece_runs",
        TensorMatcher({-1, -1, device::expert_stream::kRowPieces, row_segments, 2})
            .with_dtype<int32_t>()
            .with_device(device),
        piece_runs);
    expert_stream::verify_named(
        "fault",
        TensorMatcher({device::expert_stream::kStreamFaultWords}).with_dtype<int32_t>().with_device(device),
        fault);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
      RuntimeCheck(lease_p % kLeaseBlockAlign == 0, "lease_p: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = StreamParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .dst_slots = static_cast<const int32_t*>(dst_slots.data_ptr()),
        .row = row,
        .lanes = lanes,
        .experts = experts,
        .host_rows_2 = static_cast<int64_t*>(host_rows_2.data_ptr()),
        .dst_slots_2 = static_cast<int32_t*>(dst_slots_2.data_ptr()),
        .ram_miss = static_cast<int64_t*>(ram_miss.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
        .lease_p = lease_p,
        .claimed = static_cast<const int32_t*>(claimed.data_ptr()),
        .go_2 = static_cast<int32_t*>(go_2.data_ptr()),
        .lane_ctx_2 = static_cast<int64_t*>(lane_ctx_2.data_ptr()),
        .origin_2 = static_cast<int32_t*>(origin_2.data_ptr()),
        .stream_count = static_cast<uint32_t*>(stream_count.data_ptr()),
        .stream_abort = static_cast<int32_t*>(stream_abort.data_ptr()),
        .segments = static_cast<const int64_t*>(segments.data_ptr()),
        .segment_count = segments.size(0),
        .segment_map = static_cast<const int32_t*>(segment_map.data_ptr()),
        .row_segments = row_segments,
        .piece_runs = static_cast<const int32_t*>(piece_runs.data_ptr()),
        .fault = static_cast<const int32_t*>(fault.data_ptr()),
    };
    LaunchKernel(device::expert_stream::kStreamBlocks, device::expert_stream::kStreamThreads, stream)(
        exl3_ram_miss_lease_stream_kernel, params);
  }

  static void lease_copy_wait(
      tvm::ffi::TensorView page,
      tvm::ffi::TensorView state,
      tvm::ffi::TensorView count,
      int64_t lease_address,
      int64_t lease_c,
      int64_t lease_d,
      int64_t sm_table_address,
      int64_t sm_count,
      tvm::ffi::TensorView go_ce) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();

    expert_stream::verify_named("page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device(on_host), page);
    expert_stream::verify_named(
        "state", TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device(device), state);
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device(device), count);
    expert_stream::verify_named("go_ce", TensorMatcher({1}).with_dtype<int32_t>().with_device(device), go_ce);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
      RuntimeCheck(lease_c % kLeaseBlockAlign == 0, "lease_c: must be a multiple of kLeaseBlockAlign");
    }
    RuntimeCheck(
        sm_count <= static_cast<int64_t>(std::popcount(L::kSmallMask)),
        "sm_table: sm_count must not exceed the layout's small-tensor count");
    RuntimeCheck(
        sm_count == 0 || sm_table_address != 0, "sm_table: address must be nonzero when sm_count > 0");

    const auto stream = LaunchKernel::resolve_device(state.device());
    const int threads = sm_count > 0 ? device::expert_stream::kCopyWaitThreads : device::expert_stream::kBlock;
    const auto params = CopyWaitParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_c = lease_c,
        .lease_d = lease_d,
        .sm_table = reinterpret_cast<const int64_t*>(sm_table_address),
        .sm_count = sm_count,
        .go_ce = static_cast<int32_t*>(go_ce.data_ptr()),
    };
    LaunchKernel(1, threads, stream)(exl3_ram_miss_lease_copy_wait_kernel, params);
  }
};

}  // namespace sglang
