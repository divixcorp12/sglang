// Row-copy kernels: S, the stream kernel, and the copy wait CW / CC (analysis/dsv41-drive/LEASE_PROTOCOL.md).
#pragma once

#include <sgl_kernel/tensor.h>

#include "lease_device.cuh"
#include "row_layout.h"
#include "tensor_checks.h"
#include <bit>
#include <dlfcn.h>

namespace sglang {

namespace device::expert_stream {

// S: kStreamBlocks blocks of kStreamThreads.
constexpr int kStreamBlocks = 8;
constexpr int kStreamThreads = 256;
constexpr int kRowPieces = 8;  // the host's kPieces: one readiness bit per piece of a row
constexpr uint32_t kAllPieces = 255u;

// ld.global.cv, never .nc: a LOADING lane's host bytes are written while the kernel runs, and .nc may serve a line
// cached before its piece was published. __ldcv/__stcg emit ld.global.cv/st.global.cg with a "memory" clobber, so
// they stay ordered after the acquire that admitted the lane.
SGL_DEVICE void stream_copy16(const uint8_t* src, uint8_t* dst) {
  __stcg(reinterpret_cast<longlong2*>(dst), __ldcv(reinterpret_cast<const longlong2*>(src)));
}

SGL_DEVICE void stream_copy1(const uint8_t* src, uint8_t* dst) {
  *dst = __ldcv(src);
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
    for (int64_t b = units * 16 + tid; b < bytes; b += kStreamThreads)
      stream_copy1(src + b, dst + b);
  }
}

// This block's slice of piece `piece` of one lane. `runs` is the lane's [kRowPieces][row_segments][2] table of
// name-row byte ranges; `segment_map` gives each row segment's copy-table entry (-1: none), then a flag per entry
// that no row segment names, which is copied whole with piece 0.
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
  int32_t mine[kLeaseLanes];      // planned and not claimed by W1: this kernel's lane
  int32_t admitted[kLeaseLanes];  // its RowResult was read for G
  int32_t loading[kLeaseLanes];   // admitted under tag LOADING (else READY: every piece is there)
  int32_t slot[kLeaseLanes];
  uint32_t done[kLeaseLanes];  // pieces this block has copied its slice of
  uint32_t todo[kLeaseLanes];  // pieces to copy this pass
  int served;                  // demand_done >= seq, and the re-read after it found every mask full
  int finished;                // served and every piece of every lane copied
};

// One lane's admission: READY or LOADING makes it this kernel's, COPYING or CPU hands it to CW (W1's budget ran out
// before it was published), unpublished leaves it for a later pass.
SGL_DEVICE void stream_admit(StreamLanes& sh, int lane, const uint8_t* results, uint64_t generation, uint32_t capacity) {
  const LaneRead r = lane_read(results + lane * kLeaseRowResultBytes, generation, capacity);
  if (copy_owned(r.tag)) {
    sh.mine[lane] = 0;
  } else if (r.tag == kLeaseTagReady || r.tag == kLeaseTagLoading) {
    sh.slot[lane] = r.host_slot;
    sh.loading[lane] = r.tag == kLeaseTagLoading ? 1 : 0;
    sh.admitted[lane] = 1;
  }
}

SGL_DEVICE uint32_t piece_bits(uint64_t word, uint64_t generation) {
  return (word >> 8) == (generation & kGenerationMask) ? static_cast<uint32_t>(word & 0xFFu) : 0u;
}

}  // namespace device::expert_stream

// S, the stream kernel. It covers every planned lane W1 did not claim, admits each once its RowResult is published
// (READY or LOADING), and copies each piece as its bit appears in the lane's PieceMask word, so the copy overlaps the
// read. Every block runs the same leader loop on its own and copies its own slice of every piece; there is no
// inter-block barrier, so no co-residency is assumed, and a block that finds the request broken traps the grid.
struct StreamParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  const int32_t* dst_slots;
  int64_t row;
  int64_t experts;
  const uint8_t* lease;
  const int32_t* claimed;
  const int64_t* segments;
  int64_t segment_count;
  const int32_t* segment_map;
  int64_t row_segments;
  const int32_t* piece_runs;
  uint32_t row_capacity;  // the row's pinned slots: fixed for the process, so a captured argument is sound
};

template <bool kUsePDL>
__global__ __launch_bounds__(device::expert_stream::kStreamThreads, 1) void exl3_ram_miss_lease_stream_kernel(
    const __grid_constant__ StreamParams p) {
  // PDL as the post kernel (lease_kernels.cuh): its primary is C1, a plain launch, whose primary is W1.
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  using namespace device::expert_stream;
  __shared__ StreamLanes sh;
  const int tid = threadIdx.x;
  // W1 trapped on a count past kLeaseLanes or the plan's buffers, and returned for an unarmed post.
  const int64_t planned_count = max(static_cast<int64_t>(p.count[0]), static_cast<int64_t>(0));
  if (planned_count == 0) return;
  const uint32_t seq = static_cast<uint32_t>(p.state[kPending]);
  const uint64_t generation = pending_generation(p.state);
  const int64_t idx = ring_index(seq);
  if (tid < kLeaseLanes) {
    sh.mine[tid] = tid < planned_count && p.claimed[tid] == 0 ? 1 : 0;
    sh.admitted[tid] = 0;
    sh.loading[tid] = 0;
    sh.slot[tid] = 0;
    sh.done[tid] = 0;
    sh.todo[tid] = 0;
  }
  if (tid == 0) {
    sh.served = 0;
    sh.finished = 0;
  }
  __syncthreads();

  const uint8_t* results = p.lease + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
  const uint8_t* masks = p.lease + kLeasePieceMask + idx * kLeaseLanes * kLeasePieceMaskLineBytes;
  const uint64_t deadline = load_deadline(p.state);
  const int64_t piece_stride = static_cast<int64_t>(kRowPieces) * p.row_segments * 2;

  while (sh.finished == 0) {
    // Admission and masks: one leader-warp lane per request lane. A READY lane (a hit W1 missed) has every piece.
    if (tid < kLeaseLanes && sh.mine[tid] != 0) {
      if (sh.admitted[tid] == 0) stream_admit(sh, tid, results, generation, p.row_capacity);
      if (sh.admitted[tid] != 0) {
        const uint32_t bits = sh.loading[tid] != 0
                                  ? piece_bits(ld_acquire_sys64(masks + tid * kLeasePieceMaskLineBytes), generation)
                                  : kAllPieces;
        sh.todo[tid] = bits & ~sh.done[tid];
      }
    }
    __syncthreads();
    bool copied = false;
    for (int lane = 0; lane < kLeaseLanes; ++lane) {
      const uint32_t todo = sh.todo[lane];
      if (todo == 0) continue;
      copied = true;
      const int32_t* runs = p.piece_runs + (p.row * p.experts + p.planned[lane]) * piece_stride;
      for (int piece = 0; piece < kRowPieces; ++piece) {
        if ((todo >> piece & 1u) == 0) continue;
        stream_copy_piece(
            p.segments, p.segment_count, p.segment_map, p.row_segments, runs, piece, sh.slot[lane], p.dst_slots[lane]);
      }
    }
    __syncthreads();
    if (tid == 0) {
      for (int lane = 0; lane < kLeaseLanes; ++lane) {
        sh.done[lane] |= sh.todo[lane];
        sh.todo[lane] = 0;
      }
      if (sh.served == 0) {
        // The one device spin nothing else bounds (a service that died, an admission that stopped): trap, so the
        // process fails instead of hanging decode.
        if (static_cast<int64_t>(global_ns() - deadline) >= 0) __trap();
        if (reached(ld_acquire_sys(p.page + kDemandDone), seq)) {
          // demand_done is stored after serve() returned, so after every grant and every PieceMask publish of G; its
          // acquire orders this thread's re-reads below after them. A mask read earlier in the pass may predate the
          // last publish, so the judgement is on these re-reads. The other threads' copies of what they admit follow
          // through the block barrier at the end of the pass.
          for (int lane = 0; lane < kLeaseLanes; ++lane) {
            if (sh.mine[lane] == 0) continue;
            if (sh.admitted[lane] == 0) stream_admit(sh, lane, results, generation, p.row_capacity);
            if (sh.mine[lane] == 0) continue;  // admitted as a COPYING or CPU lane just now
            if (sh.admitted[lane] == 0) __trap();  // served, and the lane was never granted: a lapped request
            const uint32_t bits = sh.loading[lane] != 0
                                      ? piece_bits(ld_acquire_sys64(masks + lane * kLeasePieceMaskLineBytes), generation)
                                      : kAllPieces;
            if (bits != kAllPieces) __trap();  // served with a piece never published
          }
          sh.served = 1;
        } else if (!copied) {
          __nanosleep(256);
        }
      }
      // Once served there is no deadline: the masks stay final until the ring slot's reuse re-initialises them,
      // 16 requests later, which cannot happen while this request is still in the chain.
      if (sh.served != 0) {
        bool all = true;
        for (int lane = 0; lane < kLeaseLanes; ++lane) {
          if (sh.mine[lane] != 0 && sh.done[lane] != kAllPieces) all = false;
        }
        if (all) sh.finished = 1;
      }
    }
    __syncthreads();
  }
}

// CW's SM reads: `bytes` of the pinned slot into the destination, four 16-byte units in flight per thread
// (ld.global.cv, as S: never .nc on host bytes). Every load has returned once its store is issued.
SGL_DEVICE void copy_wait_read(const uint8_t* src, uint8_t* dst, int64_t bytes) {
  const bool aligned = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0;
  const int64_t units = aligned ? bytes / 16 : 0;
  const int64_t step = blockDim.x;
  int64_t u = threadIdx.x;
  for (; u + 3 * step < units; u += 4 * step) {
    longlong2 v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      v[k] = __ldcv(reinterpret_cast<const longlong2*>(src + 16 * (u + k * step)));
#pragma unroll
    for (int k = 0; k < 4; ++k)
      __stcg(reinterpret_cast<longlong2*>(dst + 16 * (u + k * step)), v[k]);
  }
  for (; u < units; u += step)
    device::expert_stream::stream_copy16(src + 16 * u, dst + 16 * u);
  for (int64_t b = units * 16 + threadIdx.x; b < bytes; b += step)
    device::expert_stream::stream_copy1(src + b, dst + b);
}

// The copy wait, the chain's tail: CW, then cuStreamWaitValue32 on the gate, then CC. Nothing spins on an SM.
// CW reads the lanes' tags once; with `sm_count` > 0 the whole block reads the `sm_count` entries of `sm_table`
// ({source slab, destination tensor, row bytes}) of every COPYING lane from its leased slot. Then it publishes
// Done = G, and, when the request has COPYING or CPU lanes, closes the gate for G and leaves them in `ce_mask`.
struct CopyWaitParams {
  int32_t* state;
  const int32_t* count;
  uint8_t* lease;
  const int64_t* sm_table;
  int64_t sm_count;
  int32_t* ce_mask;  // the armed lanes for CC: COPYING and CPU in the low byte, CPU again in the next; 0: none
};

struct CopyCommitParams {
  const int32_t* state;
  const uint8_t* lease;
  const int32_t* ce_mask;
  int32_t* cpu_lanes;  // CPU experts: the lanes the CPU computed (a lane mask), else 0; null when off
};

constexpr int kCeMaskCpuShift = 8;

template <bool kUsePDL>
__global__ __launch_bounds__(device::expert_stream::kCopyWaitThreads, 1) void exl3_ram_miss_lease_copy_wait_kernel(
    const __grid_constant__ CopyWaitParams p) {
  // PDL as the post kernel (lease_kernels.cuh). Its primary is S, so C1 and S have completed before any read below.
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  using namespace device::expert_stream;
  __shared__ int32_t sm_host[kLeaseLanes];
  __shared__ int32_t sm_dst[kLeaseLanes];
  __shared__ uint32_t copying;  // COPYING lanes: their DMA and, with sm_count, these reads fill the slot
  __shared__ uint32_t cpu;      // CPU lanes: the CPU expert thread computes them
  const int64_t planned_count = max(static_cast<int64_t>(p.count[0]), static_cast<int64_t>(0));
  const uint32_t seq = static_cast<uint32_t>(p.state[kPending]);
  const uint64_t generation = pending_generation(p.state);
  const int64_t idx = ring_index(seq);
  uint8_t* const lease = p.lease;
  if (threadIdx.x == 0) {
    uint32_t c = 0, u = 0;
    if (planned_count > 0) {
      // Every lane of G was granted before demand_done, which S acquired: these tags are final.
      const uint8_t* results = lease + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
      const uint8_t* request = lease + kLeaseLaneRequest + idx * kLeaseLaneRequestBytes;
      for (int64_t lane = 0; lane < planned_count; ++lane) {
        const uint64_t word = ld_acquire_sys64(results + lane * kLeaseRowResultBytes + kLeaseRrReady);
        if ((word & kGenerationMask) != generation) continue;
        if ((word >> 56) == kLeaseTagCpu) u |= 1u << lane;
        if ((word >> 56) != kLeaseTagCopying) continue;
        c |= 1u << lane;
        // After the acquire of the ready word; a COPYING lane's payload is fixed until its lease is released.
        sm_host[lane] = ld_relaxed_sys<int32_t>(results + lane * kLeaseRowResultBytes + kLeaseRrHostSlot);
        sm_dst[lane] = ld_relaxed_sys<int32_t>(request + kLeaseLrDst + 4 * lane);
      }
    }
    copying = c;
    cpu = u;
  }
  __syncthreads();
  if (p.sm_count > 0) {
#ifdef EXL3_RAM_MISS_TEST_CW_SM_READ_DELAY_NS
    // Test only (device_module_with_hooks): every warp but the first reads late, so a Done ahead of any read shows.
    if (threadIdx.x >= 32) {
      const uint64_t start = global_ns();
      while (global_ns() - start < static_cast<uint64_t>(EXL3_RAM_MISS_TEST_CW_SM_READ_DELAY_NS)) {
      }
    }
#endif
    for (uint32_t lanes = copying; lanes != 0; lanes &= lanes - 1) {
      const int lane = __ffs(lanes) - 1;
      for (int64_t k = 0; k < p.sm_count; ++k) {
        const int64_t* e = p.sm_table + 3 * k;
        copy_wait_read(
            reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(e[0])) + sm_host[lane] * e[2],
            reinterpret_cast<uint8_t*>(static_cast<intptr_t>(e[1])) + sm_dst[lane] * e[2],
            e[2]);
      }
    }
  }
  __syncthreads();  // every thread's SM reads before thread 0's Done
  if (threadIdx.x != 0) return;
  p.ce_mask[0] = 0;
  if (planned_count == 0) return;
  // Done: C1's and S's reads of G's leased slots finished with their kernels, before this one started (the stream
  // order, or PDL's wait, which is transitive), and this block's SM reads before the barrier above. The fence makes
  // them performed at system scope before the host can see Done and release a slot for rewriting.
  __threadfence_system();
  st_release_sys64(lease + kLeaseDone + idx * kLeaseDoneBytes, generation);
  const uint32_t mask = copying | cpu;
  if (mask == 0) return;
  uint8_t* const gate = lease + kLeaseCopyGate;
  st_relaxed_sys<uint32_t>(gate, copy_gate_word(seq, kLeaseGateClosed));
  p.ce_mask[0] = static_cast<int32_t>(mask | cpu << kCeMaskCpuShift);
  // Dekker with the copy thread (RamTier::copy_completed: CopyDone store, fence, gate load): this close is ordered
  // before the CopyDone load below, so one side always sees the other's store and opens the gate. Both open with the
  // same word, so opening twice is harmless.
  __threadfence_system();
  if (ld_acquire_sys64(lease + kLeaseCopyDone + idx * kLeaseCopyDoneBytes) == generation) {
    st_release_sys(gate, copy_gate_word(seq, kLeaseGateOpen));
  }
}

// CC, after the stream wait on the gate: a plain launch (the wait node before it is not a kernel). The gate is only
// the wake-up; CopyDone == G is what commits. The copy thread stored it after the DMA and the CPU forward completed,
// and its acquire here orders the CPU's partial sums, which the fused MoE reads next, after them.
__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_lease_copy_commit_kernel(
    const __grid_constant__ CopyCommitParams p) {
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  const uint32_t armed = static_cast<uint32_t>(p.ce_mask[0]);
  if (p.cpu_lanes != nullptr) p.cpu_lanes[0] = 0;
  if ((armed & 0xFFu) == 0) return;
  const uint32_t seq = static_cast<uint32_t>(p.state[kPending]);
  const uint64_t generation = pending_generation(p.state);
  // Only a teardown opens a gate without CopyDone: the service is gone, and the copies may not have landed.
  if (ld_acquire_sys64(p.lease + kLeaseCopyDone + ring_index(seq) * kLeaseCopyDoneBytes) != generation) __trap();
  if (p.cpu_lanes != nullptr) p.cpu_lanes[0] = static_cast<int32_t>(armed >> kCeMaskCpuShift & 0xFFu);
}

namespace expert_stream {

// cuStreamWaitValue32_v2 from libcuda.so.1, resolved once. The v2 name, never the plain one: that is the v1 API,
// gated by NVreg_EnableStreamMemOPs (copy_engine.h). Null when the driver lacks it; the launcher then refuses.
using StreamWaitValue32 = int (*)(void*, uint64_t, uint32_t, unsigned);
inline StreamWaitValue32 stream_wait_value32() {
  static const StreamWaitValue32 fn = [] {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr) lib = dlopen("libcuda.so.1", RTLD_NOW);
    return lib == nullptr ? nullptr : reinterpret_cast<StreamWaitValue32>(dlsym(lib, "cuStreamWaitValue32_v2"));
  }();
  return fn;
}
constexpr unsigned kStreamWaitValueGeq = 0;  // CU_STREAM_WAIT_VALUE_GEQ: (int32_t)(*addr - value) >= 0

}  // namespace expert_stream

/// \brief Checked host launchers for the row-copy kernels above (lease_stream, lease_copy_wait), templated on the
/// streamed row's compile-time layout facts (name count, small-tensor mask).
/// \tparam L The streamed row's compile-time layout (`expert_stream::ExpertRowLayout`).
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
      int64_t lease_address,
      tvm::ffi::TensorView claimed,
      tvm::ffi::TensorView segments,
      tvm::ffi::TensorView segment_map,
      int64_t row_segments,
      tvm::ffi::TensorView piece_runs,
      int64_t row_capacity,
      int64_t use_pdl) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto P_ = SymbolicSize{"planned"};
    // The copy table's row count is runtime data (a test may build one with far fewer entries than the layout's
    // name count): bind it from `segments` itself and cross-check segment_map against that.
    auto S_ = SymbolicSize{"segments"};

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "planned", TensorMatcher({P_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "dst_slots", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots);
    // S reads planned and dst_slots at every lane W1 bounded by min(host_rows_1, dst_slots): kLeaseLanes at most.
    RuntimeCheck(
        P_.unwrap() >= std::min<int64_t>(kLeaseLanes, dst_slots.size(0)),
        "planned: must have at least as many lanes as dst_slots, up to kLeaseLanes");
    expert_stream::verify_named(
        "claimed", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), claimed);
    expert_stream::verify_named(
        "segments", TensorMatcher({S_, 3}).with_dtype<int64_t>().with_device<kDLCUDA>(device), segments);
    expert_stream::verify_named(
        "segment_map", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), segment_map);
    RuntimeCheck(
        segment_map.size(0) == row_segments + S_.unwrap(),
        "segment_map: size must equal row_segments + the copy table's entry count");
    expert_stream::verify_named(
        "piece_runs",
        TensorMatcher({-1, -1, device::expert_stream::kRowPieces, row_segments, 2})
            .with_dtype<int32_t>()
            .with_device<kDLCUDA>(device),
        piece_runs);
    RuntimeCheck(
        lease_address != 0 && lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be a nonzero multiple of kLeaseBlockAlign");

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = StreamParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .dst_slots = static_cast<const int32_t*>(dst_slots.data_ptr()),
        .row = row,
        .experts = experts,
        .lease = reinterpret_cast<const uint8_t*>(lease_address),
        .claimed = static_cast<const int32_t*>(claimed.data_ptr()),
        .segments = static_cast<const int64_t*>(segments.data_ptr()),
        .segment_count = segments.size(0),
        .segment_map = static_cast<const int32_t*>(segment_map.data_ptr()),
        .row_segments = row_segments,
        .piece_runs = static_cast<const int32_t*>(piece_runs.data_ptr()),
        .row_capacity = expert_stream::checked_row_capacity(row_capacity),
    };
    LaunchKernel(device::expert_stream::kStreamBlocks, device::expert_stream::kStreamThreads, stream).enable_pdl(use_pdl != 0)(
        use_pdl != 0 ? exl3_ram_miss_lease_stream_kernel<true> : exl3_ram_miss_lease_stream_kernel<false>, params);
  }

  static void lease_copy_wait(
      tvm::ffi::TensorView state,
      tvm::ffi::TensorView count,
      int64_t lease_address,
      int64_t sm_table_address,
      int64_t sm_count,
      tvm::ffi::TensorView ce_mask,
      tvm::ffi::TensorView cpu_lanes,
      int64_t use_pdl) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "ce_mask", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), ce_mask);
    // CPU experts off: an empty tensor, and CC writes no CPU lanes.
    expert_stream::verify_named(
        "cpu_lanes", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), cpu_lanes);
    RuntimeCheck(cpu_lanes.size(0) <= 1, "cpu_lanes: one word, or empty when CPU experts are off");
    RuntimeCheck(
        lease_address != 0 && lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be a nonzero multiple of kLeaseBlockAlign");
    RuntimeCheck(
        expert_stream::stream_wait_value32() != nullptr,
        "the copy wait needs cuStreamWaitValue32_v2, which libcuda.so.1 does not provide");
    RuntimeCheck(
        sm_count <= static_cast<int64_t>(std::popcount(L::kSmallMask)),
        "sm_table: sm_count must not exceed the layout's small-tensor count");
    RuntimeCheck(sm_count == 0 || sm_table_address != 0, "sm_table: address must be nonzero when sm_count > 0");

    const auto stream = LaunchKernel::resolve_device(state.device());
    const int threads = sm_count > 0 ? device::expert_stream::kCopyWaitThreads : device::expert_stream::kBlock;
    const auto params = CopyWaitParams{
        .state = static_cast<int32_t*>(state.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .sm_table = reinterpret_cast<const int64_t*>(sm_table_address),
        .sm_count = sm_count,
        .ce_mask = static_cast<int32_t*>(ce_mask.data_ptr()),
    };
    LaunchKernel(1, threads, stream).enable_pdl(use_pdl != 0)(
        use_pdl != 0 ? exl3_ram_miss_lease_copy_wait_kernel<true> : exl3_ram_miss_lease_copy_wait_kernel<false>, params);
    // The gate is not a counter, so the cyclic GEQ never wraps: an open word (29-bit seq << 2 | 1) is in [1, 2^31)
    // and passes, a closed word has bit 31 set and blocks. Captured as a memory-op node of the graph.
    const int r = expert_stream::stream_wait_value32()(
        static_cast<void*>(stream),
        static_cast<uint64_t>(lease_address + kLeaseCopyGate),
        kLeaseGateOpen,
        expert_stream::kStreamWaitValueGeq);
    RuntimeCheck(r == 0, "the copy wait's cuStreamWaitValue32_v2 on the gate failed: CUresult ", r);
    const auto commit = CopyCommitParams{
        .state = static_cast<const int32_t*>(state.data_ptr()),
        .lease = reinterpret_cast<const uint8_t*>(lease_address),
        .ce_mask = static_cast<const int32_t*>(ce_mask.data_ptr()),
        .cpu_lanes = cpu_lanes.size(0) == 1 ? static_cast<int32_t*>(cpu_lanes.data_ptr()) : nullptr,
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream)(exl3_ram_miss_lease_copy_commit_kernel, commit);
  }
};

}  // namespace sglang
