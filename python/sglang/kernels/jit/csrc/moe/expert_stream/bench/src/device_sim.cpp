#include "device_sim.h"

#include "expert_stream/lease_layout.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <time.h>

namespace fullstack {
namespace w = ::sglang::expert_stream::wire;
static_assert(kLanes == w::kLeaseLanes && kLanes == w::kMaxIds);

int64_t monotonic_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

namespace {

template <class T>
T load_acquire(const uint8_t* p) {
  return __atomic_load_n(reinterpret_cast<const T*>(p), __ATOMIC_ACQUIRE);
}

template <class T>
void store_release(uint8_t* p, T value) {
  __atomic_store_n(reinterpret_cast<T*>(p), value, __ATOMIC_RELEASE);
}

template <class T>
void put(uint8_t* p, T value) {
  std::memcpy(p, &value, sizeof(value));
}

uint32_t gate_word(uint32_t seq, uint32_t low) {
  return ((seq & w::kLeaseGateSeqMask) << w::kLeaseGateSeqShift) | low;
}

}  // namespace

DeviceSim::DeviceSim(uint8_t* page, uint8_t* lease, int64_t rows, int64_t experts, uint32_t epoch)
    : page_(page),
      lease_(lease),
      rows_(rows),
      experts_(experts),
      epoch_(epoch),
      ram_slot_(static_cast<size_t>(rows * experts), -1),
      staging_(static_cast<size_t>(rows)),
      map_chain_(static_cast<size_t>(rows), 1),
      map_applied_(static_cast<size_t>(rows), 0),
      row_cpu_(static_cast<size_t>(rows), 0) {
  for (auto& s : staging_)
    s.fill(-1);
}

void DeviceSim::set_row_cpu(int64_t row) {
  row_cpu_.at(static_cast<size_t>(row)) = 1;
}

// As ChainSim.apply_pending: the tag is acquired (pairing with the host's release), then the payload is read.
bool DeviceSim::apply_pending(int64_t row) {
  const uint8_t* d = lease_ + w::kDeltaBase + row * w::kDeltaStride;
  const uint64_t tag = load_acquire<uint64_t>(d + w::kDeltaTag);
  if (tag != map_chain_[row]) return false;
  if (map_applied_[row] == tag) return true;
  uint32_t count;
  std::memcpy(&count, d + w::kDeltaCount, 4);
  if (count > static_cast<uint32_t>(w::kDeltaMaxEntries))
    throw std::runtime_error(
        "row " + std::to_string(row) + ": delta " + std::to_string(tag) + " has " + std::to_string(count) + " entries");
  for (uint32_t i = 0; i < count; ++i) {
    int16_t entry[2];
    std::memcpy(entry, d + w::kDeltaEntries + 4 * i, 4);
    if (entry[0] < 0 || entry[0] >= experts_)
      throw std::runtime_error("a delta entry names expert " + std::to_string(entry[0]));
    ram_slot_[row * experts_ + entry[0]] = entry[1];
  }
  for (int k = 0; k < kLanes; ++k) {
    int16_t slot;
    std::memcpy(&slot, d + w::kDeltaStaging + 2 * k, 2);
    staging_[row][k] = slot;
  }
  map_applied_[row] = tag;
  return true;
}

void DeviceSim::sync_row(int64_t row, int64_t deadline_ns) {
  if (row < 0 || row >= rows_) throw std::runtime_error("row " + std::to_string(row) + " is out of range");
  for (uint32_t spin = 0; !apply_pending(row); ++spin) {
    if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns)
      throw std::runtime_error(
          "row " + std::to_string(row) + ": the host has not published delta " + std::to_string(map_chain_[row]));
    _mm_pause();
  }
}

SimRequest DeviceSim::post(
    int64_t row,
    std::span<const int32_t> experts,
    std::span<const float> weights,
    bool captured,
    int64_t deadline_ns,
    const PostHook& before_publish) {
  const int count = static_cast<int>(experts.size());
  if (row < 0 || row >= rows_) throw std::runtime_error("row " + std::to_string(row) + " is out of range");
  if (count < 1 || count > kLanes) throw std::runtime_error("a post has 1.." + std::to_string(kLanes) + " lanes");
  if (weights.size() != experts.size()) throw std::runtime_error("a post needs one of its weights per lane");
  for (int j = 0; j < count; ++j) {
    if (experts[j] < 0 || experts[j] >= experts_)
      throw std::runtime_error("expert " + std::to_string(experts[j]) + " is out of range");
    for (int i = 0; i < j; ++i)
      if (experts[i] == experts[j])
        throw std::runtime_error("a post names expert " + std::to_string(experts[j]) + " twice");
  }
  sync_row(row, deadline_ns);

  // Lane typing as ram_slot_map.type_lanes with hit_copy="sm", cpu_misses=false, cpu_ok=ce_ok=true.
  SimRequest r;
  r.row = row;
  r.count = count;
  bool hit[kLanes] = {};
  int m = 0;
  for (int j = 0; j < count; ++j) {
    r.experts[j] = experts[j];
    const int32_t slot = ram_slot(row, experts[j]);
    if (slot >= 0) {
      r.slots[j] = slot;
      hit[j] = true;
      continue;
    }
    if (m >= kLanes || staging_[row][m] < 0) throw std::runtime_error("a miss lane has no staging slot");
    r.slots[j] = staging_[row][m++];
  }
  const bool host_lanes = captured && load_acquire<uint32_t>(lease_ + w::kCopyArmed) != 0;
  bool eligible[kLanes] = {};
  int n = 0;
  for (int j = 0; j < count; ++j) {
    eligible[j] = host_lanes && row_cpu_[row] != 0 && hit[j];
    n += eligible[j] ? 1 : 0;
  }
  int take = n > 0 ? __atomic_load_n(reinterpret_cast<const int32_t*>(lease_ + w::kSplit) + n, __ATOMIC_RELAXED) : 0;
  bool cpu[kLanes] = {};
  for (int j = count - 1; j >= 0 && take > 0; --j) {
    if (eligible[j]) {
      cpu[j] = true;
      --take;
    }
  }
  bool miss = false;
  for (int j = 0; j < count; ++j) {
    r.kinds[j] = static_cast<int32_t>(cpu[j] ? w::kKindHitCpu : hit[j] ? w::kKindHitSm : w::kKindMissGpu);
    miss = miss || !hit[j];
  }
  if (miss) r.chain = ++map_chain_[row];

  const uint32_t head = __atomic_load_n(reinterpret_cast<const uint32_t*>(page_ + w::kDemandHead), __ATOMIC_RELAXED);
  uint32_t seq = head + 1;
  if (seq == 0) {  // the post kernel never posts 0: it wraps to 1 in the next epoch
    seq = 1;
    ++epoch_;
  }
  r.seq = seq;
  r.gen = static_cast<uint64_t>(epoch_) << 32 | seq;
  r.idx = static_cast<int64_t>((seq - 1) % w::kDemandRecords);
  uint8_t* rec = page_ + w::kDemandRing + r.idx * w::kRecordBytes;

  // Seqlock write: seq = 0 first. x86 keeps stores in order; the barrier keeps the compiler from hoisting the payload.
  __atomic_store_n(reinterpret_cast<uint32_t*>(rec + w::kRecSeq), 0u, __ATOMIC_RELAXED);
  asm volatile("" ::: "memory");
  put<uint16_t>(rec + w::kRecRow, static_cast<uint16_t>(row));
  put<uint8_t>(rec + w::kRecCounts, static_cast<uint8_t>(count | count << 4));  // protect ids = the lane experts
  put<uint8_t>(rec + w::kRecFlags, static_cast<uint8_t>(captured ? w::kRecFlagCaptured : 0));
  put<uint64_t>(rec + w::kRecChain, r.chain);
  put<uint32_t>(rec + w::kRecEpoch, epoch_);
  uint32_t kinds = 0;
  for (int j = 0; j < count; ++j)
    kinds |= (static_cast<uint32_t>(r.kinds[j]) & 0xFu) << (4 * j);
  put<uint32_t>(rec + w::kRecKinds, kinds);
  for (int j = 0; j < kLanes; ++j) {
    const bool lane = j < count;
    put<int16_t>(rec + w::kRecProtect + 2 * j, static_cast<int16_t>(lane ? experts[j] : -1));
    put<int16_t>(rec + w::kRecLaneExpert + 2 * j, static_cast<int16_t>(lane ? experts[j] : -1));
    put<int16_t>(rec + w::kRecLaneSlot + 2 * j, static_cast<int16_t>(lane ? r.slots[j] : -1));
    put<int16_t>(rec + w::kRecLaneDst + 2 * j, static_cast<int16_t>(lane ? j : -1));
    put<float>(rec + w::kRecLaneWeight + 4 * j, lane ? weights[j] : 0.0f);
  }
  if (before_publish) before_publish(rec);
  store_release<uint32_t>(rec + w::kRecSeq, seq);
  store_release<uint32_t>(page_ + w::kDemandHead, seq);
  return r;
}

bool DeviceSim::needs_copy_wait(const SimRequest& r) {
  for (int j = 0; j < r.count; ++j) {
    const auto kind = static_cast<uint32_t>(r.kinds[j]);
    if (kind == w::kKindHitCopy || kind == w::kKindHitCpu || kind == w::kKindMissCpu) return true;
  }
  return false;
}

bool DeviceSim::copy_wait(const SimRequest& r, int64_t deadline_ns) {
  if (!needs_copy_wait(r)) return true;
  auto* gate = reinterpret_cast<uint32_t*>(lease_ + w::kLeaseCopyGate);
  const uint32_t closed = gate_word(r.seq, w::kLeaseGateClosed);
  __atomic_store_n(gate, closed, __ATOMIC_SEQ_CST);  // close, then (seq_cst) read CopyDone
  const uint8_t* done = lease_ + w::kLeaseCopyDone + r.idx * w::kLeaseCopyDoneBytes;
  for (uint32_t spin = 0; load_acquire<uint64_t>(done) != r.gen; ++spin) {
    if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns) return false;
    _mm_pause();
  }
  // Open from G's closed word only (the host's CAS rule): nothing else opens a gate closed after CopyDone.
  uint32_t expected = closed;
  __atomic_compare_exchange_n(
      gate, &expected, gate_word(r.seq, w::kLeaseGateOpen), false, __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE);
  return true;
}

bool DeviceSim::wait_pieces(const SimRequest& r, int lane, int64_t deadline_ns) const {
  const uint8_t* word = lease_ + w::kLeasePieceMask + (r.idx * w::kLeaseLanes + lane) * w::kLeasePieceMaskLineBytes;
  const uint64_t want = ((r.gen & ((uint64_t{1} << 56) - 1)) << 8) | 0xFFu;  // piece_word(G, every piece)
  for (uint32_t spin = 0; load_acquire<uint64_t>(word) != want; ++spin) {
    if ((spin & 1023) == 1023 && monotonic_ns() > deadline_ns) return false;
    _mm_pause();
  }
  return true;
}

int32_t DeviceSim::ram_slot(int64_t row, int32_t expert) const {
  return ram_slot_.at(static_cast<size_t>(row * experts_ + expert));
}

std::array<int32_t, kLanes> DeviceSim::staging(int64_t row) const {
  return staging_.at(static_cast<size_t>(row));
}

uint64_t DeviceSim::map_chain(int64_t row) const {
  return map_chain_.at(static_cast<size_t>(row));
}

uint64_t DeviceSim::copy_done(const SimRequest& r) const {
  return load_acquire<uint64_t>(lease_ + w::kLeaseCopyDone + r.idx * w::kLeaseCopyDoneBytes);
}

uint32_t DeviceSim::copy_gate() const {
  return load_acquire<uint32_t>(lease_ + w::kLeaseCopyGate);
}

uint32_t DeviceSim::epoch() const {
  return epoch_;
}

uint32_t load_experts(DeviceSim& sim, int64_t row, std::span<const int32_t> experts, int staging, int64_t timeout_ns) {
  if (staging < 1) throw std::runtime_error("load_experts needs at least one staging slot");
  uint32_t last = 0;
  for (size_t first = 0; first < experts.size(); first += static_cast<size_t>(staging)) {
    const auto group = experts.subspan(first, std::min<size_t>(static_cast<size_t>(staging), experts.size() - first));
    const std::vector<float> weights(group.size(), 1.0f);
    const int64_t deadline = monotonic_ns() + timeout_ns;
    const SimRequest r = sim.post(row, group, weights, /*captured=*/false, deadline);
    for (int j = 0; j < r.count; ++j) {
      if (r.kinds[j] != static_cast<int32_t>(w::kKindMissGpu))
        throw std::runtime_error(
            "row " + std::to_string(row) + ": loading expert " + std::to_string(group[j]) +
            ", which is already resident");
      if (!sim.wait_pieces(r, j, deadline))
        throw std::runtime_error(
            "row " + std::to_string(row) + ": expert " + std::to_string(group[j]) +
            "'s pieces did not land in time (gen " + std::to_string(r.gen) + ")");
    }
    last = r.seq;
  }
  sim.sync_row(row, monotonic_ns() + timeout_ns);
  for (int32_t e : experts)
    if (sim.ram_slot(row, e) < 0)
      throw std::runtime_error(
          "row " + std::to_string(row) + ": expert " + std::to_string(e) + " is not resident after its load");
  return last;
}

}  // namespace fullstack
