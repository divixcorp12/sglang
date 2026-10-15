// The GPU drip (SGLANG_DSV41_GPU_DRIP_SLOTS): copies experts the GPU scorer predicts for a later layer from their
// pinned RAM rows into a small VRAM scratch while the link carries no demand transfer, and publishes each landed row
// in the scratch table, a device-memory u64 per slot the copy stream writes after the row's bytes. The target layer's
// post types a lane whose (row, expert) a published word names kKindHitScratch, and C1s copies it from the scratch
// slot, so the lane leaves the CPU split and the demand copies.
//
//   DripConfig    slots, picks per scored record, chunk size, the scorer's targets and candidate page
//   GpuDrip       the scratch slots' state machine, driven by the copy thread (CopyEngine::run) between its jobs
//
// A slot is free, filling (its chunks are being copied), publishing (its word's store is queued), published,
// clearing (its word's zero is queued) or retiring (its word is 0, and a post that read it before may still be in its
// chain). Every state is the copy thread's; the service threads read only ident_ (holds), to check a kKindHitScratch
// lane.
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "The GPU drip".
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <span>
#include <vector>

#include "../lease_layout.h"
#include "../spec_candidates.h"
#include "copy_engine.h"
#include "job_trace.h"
#include "tier_protocol.h"

namespace sglang {
namespace expert_stream {

struct DripConfig {
  int slots = 0;           // scratch slots, 1..Wire::kScratchSlots
  int picks = 0;           // drips admitted per scored record, 1..SpecCandidates::kMaxDrips
  int64_t chunk_bytes = 0;  // the most one copy-engine mark carries: a demand transfer waits behind at most this
  std::vector<int32_t> target;  // per source row, the row its record's candidates are for (-1: none)
  const uint8_t* candidates = nullptr;  // the GPU scorer's candidate page; null: picks only from offer() (tests)
};

// One scratch tensor of the layout (one per copy-table entry, in the copy table's order): its address and the bytes
// of one expert row in it; slot s of it starts at address + s * bytes.
struct ScratchEntry {
  uint64_t address = 0;
  int64_t bytes = 0;
};

// `Owner` is the RamTier, which provides, all called on the copy thread: handled_through(), copy_armed() and
// copy_count<K>(n).
template <class Build, class Owner>
class GpuDrip {
 public:
  enum : uint8_t { kFree = 0, kFilling = 1, kPublished = 2, kRetiring = 3, kPublishing = 4, kClearing = 5 };
  // Why a drip was dropped before publishing (the drip_abandon event's a).
  enum : int { kWhyTargetPosted = 0, kWhySourceChanged = 1, kWhyLapped = 2 };

  // `page` the request page, `lease` the completion block (its copy gate), `map` the host mirror [rows, experts] and
  // `gen` its per-entry change counter (RamTier::publish_mirror), all owned by the tier, which outlives this.
  GpuDrip(DripConfig config, Owner* owner, const uint8_t* page, const uint8_t* lease, const int32_t* map,
          const std::atomic<uint32_t>* gen, int64_t rows, int64_t experts, const std::string& name)
      : config_(std::move(config)),
        owner_(owner),
        page_(page),
        lease_(lease),
        map_(map),
        gen_(gen),
        rows_(rows),
        experts_(experts),
        sources_(static_cast<size_t>(rows)),
        last_posted_(static_cast<size_t>(rows), 0),
        trace_(name) {
    awaiting_.reserve(kAwaiting);
    offers_.reserve(kOffers);
  }

  // ---- Python, before the drips start ----

  // The scratch tensors and the scratch table (device memory, kScratchSlots zeroed u64 words the post reads), once,
  // before any row's sources (set_row). Throws on an empty or repeated call.
  void set_scratch(std::vector<ScratchEntry> entries, uint64_t table) {
    if (scratch_ready_.load(std::memory_order_acquire)) throw std::runtime_error("the drip scratch is already set");
    if (entries.empty()) throw std::runtime_error("the drip scratch needs at least one tensor");
    if (table == 0 || table % 8 != 0) throw std::runtime_error("the drip's scratch table must be an aligned address");
    for (const ScratchEntry& e : entries)
      if (e.address == 0 || e.bytes <= 0) throw std::runtime_error("a drip scratch tensor of no bytes");
    scratch_ = std::move(entries);
    table_ = table;
    scratch_ready_.store(true, std::memory_order_release);
  }

  // set_copy_table's hook: row `row`'s RAM slabs, from its copy table (every entry, SM ones included). Throws unless
  // they match the scratch tensors one for one, in bytes per row.
  void set_row(int64_t row, std::span<const CopyEntry> table) {
    if (!scratch_ready_.load(std::memory_order_acquire))
      throw std::runtime_error("set the drip scratch before any copy table");
    if (row < 0 || row >= rows_) throw std::runtime_error("drip row out of range");
    RowSource& source = sources_[row];
    if (source.ready.load(std::memory_order_acquire)) throw std::runtime_error("a row's drip sources are set once");
    if (table.size() != scratch_.size())
      throw std::runtime_error(
          "row " + std::to_string(row) + "'s copy table has " + std::to_string(table.size()) + " tensors, the drip " +
          "scratch " + std::to_string(scratch_.size()));
    for (size_t k = 0; k < table.size(); ++k) {
      if (table[k].bytes != scratch_[k].bytes)
        throw std::runtime_error(
            "row " + std::to_string(row) + "'s tensor " + std::to_string(k) + " holds " +
            std::to_string(table[k].bytes) + " bytes a row, the drip scratch " + std::to_string(scratch_[k].bytes));
      source.src.push_back(table[k].src);
    }
    source.ready.store(true, std::memory_order_release);
  }

  // ---- Any thread ----

  // Test and tool hook: picks for `row` as a scored record's candidates would name them, admitted at the copy
  // thread's next step. Dropped beyond kOffers pending.
  void offer(int64_t row, std::span<const int32_t> experts) {
    std::lock_guard<std::mutex> lock(offers_mutex_);
    if (offers_.size() >= kOffers || row < 0 || row >= rows_) return;
    Offer o;
    o.row = row;
    o.count = static_cast<int>(std::min<size_t>(experts.size(), SpecCandidates::kMaxDrips));
    std::copy_n(experts.begin(), o.count, o.experts.begin());
    offers_.push_back(o);
  }

  // Test hook: while held no chunk is issued, so admitted drips stay unlanded.
  void hold(bool on) {
    held_.store(on, std::memory_order_release);
  }

  // A service thread, checking a kKindHitScratch lane: whether `slot` holds `expert`'s row of `row` for some post
  // that may have read its word. True from just before the word's store is queued until every record that could have
  // read it was handled (release_quarantined), so a lane the device typed from the word always finds it.
  bool holds(int32_t slot, int64_t row, int32_t expert) const {
    return slot >= 0 && slot < config_.slots && row >= 0 && row < rows_ && expert >= 0 && expert < experts_ &&
           ident_[slot].load(std::memory_order_acquire) == ident(row, expert);
  }

  // Test view, racy by nature: per slot {state, row, expert, RAM slot}.
  void view(int64_t* out) const {
    for (int s = 0; s < Wire::kScratchSlots; ++s) {
      const Slot& slot = slots_[s];
      out[4 * s] = slot.state;
      out[4 * s + 1] = slot.row;
      out[4 * s + 2] = slot.expert;
      out[4 * s + 3] = slot.ram_slot;
    }
  }

  // ---- The copy thread ----

  // One step between the copy engine's own polls: follows the posted records, admits picks, frees quarantined
  // slots, polls the drip's mark in flight, queues a table store if one is due, else issues the next chunk when
  // `demand_idle` (no demand copy queued or still copying) and no device pull is pending (pull_pending). One drip mark
  // at most is in flight. Returns whether anything changed.
  bool step(CopyBackend& backend, bool demand_idle) {
    bool progressed = scan();
    progressed |= take_offers();
    progressed |= admit_candidates();
    progressed |= release_quarantined();
    if (op_slot_ >= 0) progressed |= poll_op(backend);
    if (op_slot_ >= 0) return progressed;
    if (issue_table_store(backend)) return true;
    if (!demand_idle || held_.load(std::memory_order_acquire)) return progressed;
    if (pull_pending()) {
      count_pause();
      return progressed;
    }
    return issue_chunk(backend) || progressed;
  }

  // A drip mark is in flight: the copy thread keeps polling, and its backend stays up until it lands.
  bool busy() const {
    return op_slot_ >= 0;
  }

 private:
  static constexpr size_t kAwaiting = Wire::kDemandRecords;
  static constexpr size_t kOffers = 64;  // arbitrary: tests offer a handful
  static constexpr uint64_t kHeld = 1ull << 63;
  enum : uint8_t { kOpChunk = 0, kOpPublish = 1, kOpClear = 2 };

  struct Slot {
    uint8_t state = kFree;
    bool abandoned = false;      // dropped while its chunk was in flight: freed when the chunk completes
    bool target_posted = false;  // its target posted while publishing: cleared once the store landed
    int32_t row = -1;
    int32_t expert = -1;
    int32_t ram_slot = -1;
    uint32_t gen = 0;         // the source's mirror generation at admission
    uint32_t source_seq = 0;  // the scored record that picked it (0: an offer)
    uint64_t order = 0;       // admission order: the oldest filling slot copies first
    size_t entry = 0;         // the next byte to issue: scratch tensor `entry`, at `offset`
    int64_t offset = 0;
    uint32_t retire_head = 0;  // the demand head read after the cleared word landed
  };
  struct RowSource {
    std::vector<uint64_t> src;  // per scratch tensor, the row's RAM slab of it
    std::atomic<bool> ready{false};
  };
  struct Awaiting {
    uint32_t seq = 0;
    int64_t row = 0;
  };
  struct Offer {
    int64_t row = 0;
    int count = 0;
    std::array<int32_t, SpecCandidates::kMaxDrips> experts{};
  };
  using Cand = SpecCandidates;

  static uint64_t ident(int64_t row, int32_t expert) {
    return kHeld | static_cast<uint64_t>(row) << 16 | static_cast<uint32_t>(expert);
  }
  uint64_t table_word(int s) const {
    return table_ + 8 * static_cast<uint64_t>(s);
  }
  int64_t at(const Slot& slot) const {
    return static_cast<int64_t>(slot.row) * experts_ + slot.expert;
  }
  // True while the source row is still the one admitted: the expert mapped at the same RAM slot, with no mirror change
  // in between. A RAM slot's bytes change only after its expert is unmapped, so this brackets a copy that read them.
  bool source_unchanged(const Slot& slot) const {
    return gen_[at(slot)].load(std::memory_order_acquire) == slot.gen &&
           __atomic_load_n(map_ + at(slot), __ATOMIC_ACQUIRE) == slot.ram_slot;
  }
  template <Counter K>
  void count(int64_t n = 1) {
    owner_->template copy_count<K>(n);
  }

  // Reads the posted records since the last step: a record of a filling or published slot's target ends its drip,
  // and a record with lanes the device pulls over the link opens a pull window (pull_pending). A lapped or torn record
  // loses what it hid, so every drip is dropped (lose_all).
  bool scan() {
    const uint32_t head = load_acquire(page_ + Wire::kDemandHead);
    if (!scanning_) {
      scan_ = skip_zero(head + 1u);
      scanning_ = true;
      return false;
    }
    if (head == 0 || !reached(head, scan_)) return false;
    if (head - scan_ >= Wire::kDemandRecords - 2) {
      lose_all(head);
      scan_ = skip_zero(head + 1u);
      return true;
    }
    while (reached(head, scan_)) {
      const uint8_t* record = page_ + record_offset(Wire::kDemandRing, Wire::kDemandRecords, scan_);
      uint16_t row = 0;
      bool pulls = false;
      if (!read_head(record, scan_, &row, &pulls)) {
        lose_all(head);
        scan_ = skip_zero(head + 1u);
        return true;
      }
      posted(scan_, row, pulls);
      scan_ = skip_zero(scan_ + 1u);
    }
    return true;
  }

  // The row of record `seq`, and whether a lane of it is one the device pulls from host memory over the link (C1's
  // HIT_SM, S's MISS_GPU), as a seqlock read of its head; false when it was overwritten.
  static bool read_head(const uint8_t* record, uint32_t seq, uint16_t* row, bool* pulls) {
    if (load_acquire(record + Wire::kRecSeq) != seq) return false;
    *row = __atomic_load_n(reinterpret_cast<const uint16_t*>(record + Wire::kRecRow), __ATOMIC_RELAXED);
    const uint8_t counts = __atomic_load_n(record + Wire::kRecCounts, __ATOMIC_RELAXED);
    uint32_t kinds[Wire::kKindWords];
    for (int w = 0; w < Wire::kKindWords; ++w)
      kinds[w] = __atomic_load_n(reinterpret_cast<const uint32_t*>(record + Wire::kRecKinds) + w, __ATOMIC_RELAXED);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire(record + Wire::kRecSeq) != seq) return false;
    const int lanes = std::min<int>(Wire::kPackedCounts ? (counts & 0xF) : counts, Wire::kLanes);
    *pulls = false;
    for (int j = 0; j < lanes; ++j) {
      const uint32_t kind = kinds[j / 8] >> (4 * (j % 8)) & 0xFu;
      *pulls |= kind == Wire::kKindHitSm || kind == Wire::kKindMissGpu;
    }
    return true;
  }

  void posted(uint32_t seq, int64_t row, bool pulls) {
    // A later post means every earlier chain ended: only this record's pulls can still be pending.
    pull_open_ = pulls;
    pull_seq_ = seq;
    if (row >= rows_) return;  // a malformed record: the service fails it
    last_posted_[row] = seq;
    for (int s = 0; s < config_.slots; ++s) {
      Slot& slot = slots_[s];
      if (slot.row != row) continue;
      end_drip(s, kWhyTargetPosted);
    }
    if (config_.candidates != nullptr && row < static_cast<int64_t>(config_.target.size()) && config_.target[row] >= 0) {
      if (awaiting_.size() == kAwaiting) awaiting_.erase(awaiting_.begin());
      awaiting_.push_back(Awaiting{seq, row});
    }
  }

  // The device's pulls of the last posted record may still be running: S and C1 read host memory over the link until
  // the chain's copy wait, which closes the copy gate for the record only after S completed (stream order). So the
  // window closes when the gate names the record, or when a later record posts; a record whose chain never closes the
  // gate (no copy-engine or CPU lane) holds it until then.
  bool pull_pending() {
    if (!pull_open_) return false;
    const uint32_t gate = channel::gate<TargetChannel>(lease_);
    if ((gate >> Wire::kLeaseGateSeqShift & Wire::kLeaseGateSeqMask) == (pull_seq_ & Wire::kLeaseGateSeqMask)) {
      pull_open_ = false;
      return false;
    }
    return true;
  }

  // drip_paused: once per pull window that held back a chunk ready to go.
  void count_pause() {
    if (paused_seq_ == pull_seq_) return;
    for (int s = 0; s < config_.slots; ++s) {
      if (slots_[s].state == kFilling && !slots_[s].abandoned) {
        paused_seq_ = pull_seq_;
        count<kDripPaused>();
        return;
      }
    }
  }

  void lose_all(uint32_t /*head*/) {
    for (int s = 0; s < config_.slots; ++s)
      end_drip(s, kWhyLapped);
    awaiting_.clear();
    pull_open_ = false;
  }

  // The target posted (or the records were lost): a filling slot is abandoned, a published one cleared, a publishing
  // one cleared once its store landed.
  void end_drip(int s, int why) {
    Slot& slot = slots_[s];
    if (slot.state == kFilling && !slot.abandoned) abandon(s, why);
    else if (slot.state == kPublished) slot.state = kClearing;
    else if (slot.state == kPublishing) slot.target_posted = true;
  }

  // Drops a filling slot: no further chunk is issued for it (issue_chunk skips it), and it is freed at once, or when
  // its chunk in flight lands (poll_op), since the DMA still writes it. Its word was never stored, so no post can have
  // read it and no quarantine applies. Counts the bytes it will now never copy.
  void abandon(int s, [[maybe_unused]] int why) {
    Slot& slot = slots_[s];
    count<kDripAbandoned>();
    int64_t left = -slot.offset;
    for (size_t k = slot.entry; k < scratch_.size(); ++k)
      left += scratch_[k].bytes;
    count<kDripSkippedBytes>(left);
    if constexpr (Build::kMetrics) trace_.emit("drip_abandon", slot.row, 0, slot.source_seq, -1, why, s, slot.expert);
    if (op_slot_ == s) {
      slot.abandoned = true;
    } else {
      slot = Slot{};
    }
  }

  // A retiring slot is free once the service handled record retire_head + 2: its post started after retire_head + 1's
  // chain ended, so no chain reads the slot any more, and every record that could carry its lane was checked.
  bool release_quarantined() {
    bool any = false;
    uint32_t handled = 0;
    bool loaded = false;
    for (int s = 0; s < config_.slots; ++s) {
      Slot& slot = slots_[s];
      if (slot.state != kRetiring) continue;
      if (!loaded) {
        handled = owner_->handled_through();
        loaded = true;
      }
      if (!reached(handled, skip_zero(skip_zero(slot.retire_head + 1u) + 1u))) continue;
      ident_[s].store(0, std::memory_order_release);
      slot = Slot{};
      any = true;
    }
    return any;
  }

  bool take_offers() {
    std::unique_lock<std::mutex> lock(offers_mutex_, std::try_to_lock);
    if (!lock.owns_lock() || offers_.empty()) return false;
    for (const Offer& o : offers_)
      admit(o.row, 0, std::span<const int32_t>(o.experts.data(), static_cast<size_t>(o.count)));
    offers_.clear();
    return true;
  }

  enum class Read { kReady, kNotYet, kLapped };

  // The drip picks of record `seq`'s candidate slot, a seqlock read of their own seq word as read_candidates reads the
  // slot's.
  Read read_drips(uint32_t seq, std::array<int32_t, Cand::kMaxDrips>* out, int* n) const {
    const uint8_t* slot = config_.candidates + Cand::slot_offset(seq);
    const uint32_t first = load_acquire(slot + Cand::kCandDripSeq);
    if (first != seq) return first != 0 && reached(first, skip_zero(seq + 1u)) ? Read::kLapped : Read::kNotYet;
    uint8_t bytes[Cand::kCandDripEnd - Cand::kCandDripCount];
    std::memcpy(bytes, slot + Cand::kCandDripCount, sizeof(bytes));
    std::atomic_thread_fence(std::memory_order_acquire);
    asm volatile("" ::: "memory");  // the copy's plain loads must stay before the seq re-check
    if (load_acquire(slot + Cand::kCandDripSeq) != seq) return Read::kLapped;
    uint16_t count;
    std::memcpy(&count, bytes, 2);
    *n = std::min<int>(count, Cand::kMaxDrips);
    for (int i = 0; i < *n; ++i) {
      uint16_t expert;
      std::memcpy(&expert, bytes + (Cand::kCandDripExperts - Cand::kCandDripCount) + 2 * i, 2);
      (*out)[i] = expert;
    }
    return Read::kReady;
  }

  // Admits the picks of scored records whose slots are ready; a slot still open once its ring index could be reused
  // is given up.
  bool admit_candidates() {
    if (awaiting_.empty()) return false;
    const uint32_t head = load_acquire(page_ + Wire::kDemandHead);
    bool any = false;
    for (size_t i = 0; i < awaiting_.size();) {
      const Awaiting a = awaiting_[i];
      std::array<int32_t, Cand::kMaxDrips> experts{};
      int n = 0;
      const Read read = read_drips(a.seq, &experts, &n);
      if (read == Read::kNotYet && head - a.seq < Cand::kCandRecords - 2) {
        ++i;
        continue;
      }
      awaiting_.erase(awaiting_.begin() + static_cast<std::ptrdiff_t>(i));
      any = true;
      if (read == Read::kReady)
        admit(config_.target[a.row], a.seq, std::span<const int32_t>(experts.data(), static_cast<size_t>(n)));
    }
    return any;
  }

  // Takes a free slot for each pick of `row` still mapped in RAM and not already dripping, up to config_.picks. A
  // record of `row` posted after the scored one (source_seq) makes the picks late: none is taken.
  void admit(int64_t row, uint32_t source_seq, std::span<const int32_t> experts) {
    if (row < 0 || row >= rows_ || !scratch_ready_.load(std::memory_order_acquire) ||
        !sources_[row].ready.load(std::memory_order_acquire) || !owner_->copy_armed())
      return;
    // scan() ran first this step, so a target record already posted is in last_posted_.
    if (source_seq != 0 && last_posted_[row] != 0 && reached(last_posted_[row], source_seq)) {
      count<kDripLate>();
      return;
    }
    int taken = 0;
    for (const int32_t expert : experts) {
      if (taken >= config_.picks) break;
      if (expert < 0 || expert >= experts_ || dripping(row, expert)) continue;
      const int64_t entry = row * experts_ + expert;
      // Generation first: a later unmap bumps it after its map store, so a stale map read here fails at land.
      const uint32_t gen = gen_[entry].load(std::memory_order_acquire);
      const int32_t ram_slot = __atomic_load_n(map_ + entry, __ATOMIC_ACQUIRE);
      if (ram_slot < 0) continue;
      const int s = free_slot();
      if (s < 0) {
        count<kDripNoSlot>();
        return;
      }
      ++taken;
      Slot& slot = slots_[s];
      slot = Slot{};
      slot.state = kFilling;
      slot.row = static_cast<int32_t>(row);
      slot.expert = expert;
      slot.ram_slot = ram_slot;
      slot.gen = gen;
      slot.source_seq = source_seq;
      slot.order = ++order_;
      if constexpr (Build::kMetrics) trace_.emit("drip_admit", row, 0, source_seq, -1, expert, s, ram_slot);
    }
  }

  bool dripping(int64_t row, int32_t expert) const {
    for (int s = 0; s < config_.slots; ++s)
      if (slots_[s].state != kFree && slots_[s].row == row && slots_[s].expert == expert) return true;
    return false;
  }

  int free_slot() const {
    for (int s = 0; s < config_.slots; ++s)
      if (slots_[s].state == kFree) return s;
    return -1;
  }

  // Queues the store of a publishing slot's word, or of a clearing slot's 0, behind everything on the copy stream.
  // Eight bytes: not held back by demand transfers or pulls.
  bool issue_table_store(CopyBackend& backend) {
    for (int s = 0; s < config_.slots; ++s) {
      Slot& slot = slots_[s];
      if (slot.state != kPublishing && slot.state != kClearing) continue;
      uint64_t word = 0;
      if (slot.state == kPublishing) {
        tag_ = tag_ == UINT32_MAX ? 1 : tag_ + 1;
        word = static_cast<uint64_t>(tag_) << 32 | static_cast<uint64_t>(slot.row & 0xFFFF) << 16 |
               static_cast<uint32_t>(slot.expert & 0xFFFF);
      }
      if (const int r = backend.write64(table_word(s), word))
        fail_stop("GPU drip: a scratch-table store failed to issue: " + std::to_string(r));
      mark(backend, s, slot.state == kPublishing ? kOpPublish : kOpClear);
      return true;
    }
    return false;
  }

  void mark(CopyBackend& backend, int s, uint8_t op) {
    if (const int r = backend.mark(&token_)) fail_stop("GPU drip: a mark failed: " + std::to_string(r));
    op_slot_ = s;
    op_ = op;
  }

  // Issues the oldest filling slot's next chunk: up to chunk_bytes over its tensors, then one mark. Its first chunk
  // re-checks the source first.
  bool issue_chunk(CopyBackend& backend) {
    int s = -1;
    for (int i = 0; i < config_.slots; ++i)
      if (slots_[i].state == kFilling && !slots_[i].abandoned && (s < 0 || slots_[i].order < slots_[s].order)) s = i;
    if (s < 0) return false;
    Slot& slot = slots_[s];
    if (slot.entry == 0 && slot.offset == 0) {
      if (!source_unchanged(slot)) {
        abandon(s, kWhySourceChanged);
        return true;
      }
      count<kDripIssued>();
      if constexpr (Build::kMetrics) trace_.emit("drip_issue", slot.row, 0, slot.source_seq, -1, slot.expert, s);
    }
    const std::vector<uint64_t>& src = sources_[slot.row].src;
    int64_t budget = config_.chunk_bytes;
    while (budget > 0 && slot.entry < scratch_.size()) {
      const ScratchEntry& e = scratch_[slot.entry];
      const int64_t bytes = std::min(budget, e.bytes - slot.offset);
      const uint64_t dst = e.address + static_cast<uint64_t>(s) * e.bytes + slot.offset;
      const uint64_t from = src[slot.entry] + static_cast<uint64_t>(slot.ram_slot) * e.bytes + slot.offset;
      if (const int r = backend.issue(dst, from, bytes)) fail_stop("GPU drip: a scratch copy failed to issue: " + std::to_string(r));
      budget -= bytes;
      slot.offset += bytes;
      if (slot.offset == e.bytes) {
        ++slot.entry;
        slot.offset = 0;
      }
    }
    mark(backend, s, kOpChunk);
    return true;
  }

  bool poll_op(CopyBackend& backend) {
    const int state = backend.query(token_);
    if (state == CopyBackend::kPending) return false;
    if (state != CopyBackend::kDone) fail_stop("GPU drip: a scratch copy failed: " + std::to_string(state));
    const int s = op_slot_;
    op_slot_ = -1;
    Slot& slot = slots_[s];
    switch (op_) {
      case kOpChunk:
        if (slot.abandoned) {
          slot = Slot{};
        } else if (slot.entry == scratch_.size()) {
          land(s);
        }
        break;
      case kOpPublish:
        slot.state = slot.target_posted ? kClearing : kPublished;
        if constexpr (Build::kMetrics) trace_.emit("drip_land", slot.row, 0, slot.source_seq, -1, slot.expert, s, tag_);
        break;
      default:
        retire(s);
    }
    return true;
  }

  // Every chunk of slot s landed: queues its word's store if the source never changed during the copy. ident_ is set
  // first, so the service finds the slot for any record whose post saw the word.
  void land(int s) {
    Slot& slot = slots_[s];
    if (!source_unchanged(slot)) {
      abandon(s, kWhySourceChanged);
      return;
    }
    ident_[s].store(ident(slot.row, slot.expert), std::memory_order_release);
    slot.state = kPublishing;
    count<kDripLanded>();
  }

  // The slot's cleared word has landed, so a post starting from now reads 0. A post that read the word before can
  // still be in its chain (only the post after the next one is known to start after this one's chain ended), so the
  // slot stays the drip's until every record up to the head read here plus one was handled (release_quarantined).
  void retire(int s) {
    Slot& slot = slots_[s];
    slot.retire_head = load_acquire(page_ + Wire::kDemandHead);
    slot.state = kRetiring;
    if constexpr (Build::kMetrics) trace_.emit("drip_retire", slot.row, 0, slot.retire_head, -1, slot.expert, s);
  }

  const DripConfig config_;
  Owner* const owner_;
  const uint8_t* const page_;
  const uint8_t* const lease_;
  const int32_t* const map_;
  const std::atomic<uint32_t>* const gen_;
  const int64_t rows_;
  const int64_t experts_;
  std::vector<ScratchEntry> scratch_;  // written once by set_scratch before scratch_ready_'s release
  uint64_t table_ = 0;                 // likewise
  std::atomic<bool> scratch_ready_{false};
  std::vector<RowSource> sources_;
  std::array<std::atomic<uint64_t>, Wire::kScratchSlots> ident_{};  // the service threads' view (holds)
  std::atomic<bool> held_{false};
  std::mutex offers_mutex_;
  std::vector<Offer> offers_;
  // The copy thread's alone from here on.
  std::array<Slot, Wire::kScratchSlots> slots_{};
  std::vector<uint32_t> last_posted_;  // per row, its last posted record's seq (0: none seen)
  std::vector<Awaiting> awaiting_;     // scored records whose candidate slots are not read yet
  bool scanning_ = false;
  uint32_t scan_ = 0;       // the next seq whose head scan() reads
  bool pull_open_ = false;  // the last posted record, pull_seq_, has device pulls that may still run
  uint32_t pull_seq_ = 0;
  uint32_t paused_seq_ = 0;  // the pull window drip_paused last counted
  int op_slot_ = -1;         // the slot whose drip mark is in flight (-1: none), its kind, and the mark
  uint8_t op_ = kOpChunk;
  int64_t token_ = 0;
  uint64_t order_ = 0;
  uint32_t tag_ = 0;
  [[no_unique_address]] JobTrace<Build::kMetrics> trace_;
};

}  // namespace expert_stream
}  // namespace sglang
