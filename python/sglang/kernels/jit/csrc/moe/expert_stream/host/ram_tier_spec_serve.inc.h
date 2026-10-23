// RamTier private members: the speculative pool and its jobs, publication, and the record path (serve_record).
// A class-body fragment: ram_tier.h includes it inside RamTier, at the place these members always sat. Include
// ram_tier.h, never this file.
#pragma once

  // A candidate slot as read_candidates finds it: ready, not yet (an older record's, or open: seq word 0), or lapped (a
  // later record's, also when it changed across the copy).
  enum class CandRead { kReady, kNotYet, kLapped };
  struct CandSlot {
    int count = 0;
    int32_t expert[wire::SpecCandidates::kMaxCandidates];
    int32_t rank[wire::SpecCandidates::kMaxCandidates];
    float margin[wire::SpecCandidates::kMaxCandidates];
  };

  // Copies `seq`'s slot as read_gpu_hot copies a hot record: acquire the seq, copy, acquire fence, re-check. A count
  // above kMaxCandidates is clamped (the kernel writes none).
  CandRead read_candidates(uint32_t seq, CandSlot* out) const {
    using Cand = wire::SpecCandidates;
    const uint8_t* slot = spec_->config.candidates + Cand::slot_offset(seq);
    const uint32_t first = load_acquire(slot + Cand::kCandSeq);
    if (first != seq) return first != 0 && reached(first, skip_zero(seq + 1u)) ? CandRead::kLapped : CandRead::kNotYet;
    uint8_t bytes[Cand::kCandPayloadBytes];
    std::memcpy(bytes, slot, sizeof(bytes));
    std::atomic_thread_fence(std::memory_order_acquire);
    asm volatile("" ::: "memory");  // the copy's plain loads must stay before the seq re-check
    if (load_acquire(slot + Cand::kCandSeq) != seq) return CandRead::kLapped;
    uint16_t count;
    std::memcpy(&count, bytes + Cand::kCandCount, 2);
    out->count = std::min<int>(count, Cand::kMaxCandidates);
    for (int i = 0; i < out->count; ++i) {
      const uint8_t* entry = bytes + Cand::kCandEntries + i * Cand::kCandEntryBytes;
      uint16_t expert;
      std::memcpy(&expert, entry + Cand::kCandExpert, 2);
      out->expert[i] = expert;
      out->rank[i] = entry[Cand::kCandRank];
      std::memcpy(&out->margin[i], entry + Cand::kCandMargin, 4);
    }
    return CandRead::kReady;
  }

  // Polls `seq`'s slot until it is ready or lapped: _mm_pause for kCandSpinNs, then kCandSleepNs sleeps, up to
  // kCandWaitNs. kNotYet when it never became ready.
  CandRead await_candidates(uint32_t seq, CandSlot* out) const {
    const int64_t start = now_ns();
    for (;;) {
      const CandRead read = read_candidates(seq, out);
      if (read != CandRead::kNotYet) return read;
      const int64_t waited = now_ns() - start;
      if (waited >= kCandWaitNs) return CandRead::kNotYet;
      if (waited < kCandSpinNs)
        _mm_pause();
      else
        std::this_thread::sleep_for(std::chrono::nanoseconds(kCandSleepNs));
    }
  }

  // serve_spec_job with the GPU scorer: waits for the record's slot, then reads, in the GPU's order, the first
  // per_layer candidates at or over the target row's margin floor, still unmapped here and not pooled before, its own
  // group's only. Both groups read the same slot and pass the same filters, so the layer's budget holds over both. The
  // GPU skipped the hot ones.
  void serve_gpu_job(int g, const SpecJob& job, int64_t target, bool threaded) {
    SpecGroup& spec = *spec_->groups[g];
    int64_t start = 0;
    if constexpr (Build::kMetrics) start = now_ns();
    CandSlot slot;
    const CandRead read = await_candidates(job.seq, &slot);
    if constexpr (Build::kMetrics) {
      stats_.add(kSpecScored);
      stats_.add(kSpecScoreNs, now_ns() - start);
    }
    if (read == CandRead::kNotYet) {
      spec_count<kSpecLate>(spec);
      return;
    }
    if (read == CandRead::kLapped || slot.count == 0) {
      spec_count<kSpecDropped>(spec);
      return;
    }
    const std::vector<float>& floors = spec_->config.min_margin;
    const float floor = floors.empty() ? -std::numeric_limits<float>::infinity() : floors[target];
    int taken = 0;
    for (int i = 0; i < slot.count && taken < spec_->config.per_layer; ++i) {
      const int32_t expert = slot.expert[i];
      if (!(slot.margin[i] >= floor) || expert >= experts_ || __atomic_load_n(map_ + target * experts_ + expert, __ATOMIC_ACQUIRE) >= 0 ||
          pool_->pooled_before(target, expert, job.seq))
        continue;
      ++taken;
      if (Wire::home(expert) != g) continue;
      if (threaded && !spec_wait_unheld(spec)) return;  // a quiescer waits for one row, not the whole job
      uint64_t pick = 0;  // spec_submit's gen: rank << 32 | the margin's fp32 bits
      if constexpr (Build::kMetrics)
        pick = (static_cast<uint64_t>(slot.rank[i]) << 32) | std::bit_cast<uint32_t>(slot.margin[i]);
      spec_read(g, job.row, target, job.seq, expert, pick);
    }
  }

  // Reads `expert`'s row of `target` into a pool entry of group g. On the shared reader it reads under the group's
  // reader turn, so a demand read waits for at most this one row; on an idle-drive reader (spec_reader) it takes no
  // turn and reads only from drives no demand reads (IdleRoots). Dropped when stale, or when the expert was mapped or
  // pooled meanwhile.
  void spec_read(int g, int64_t source, int64_t target, uint32_t source_seq, int32_t expert,
                 [[maybe_unused]] uint64_t pick = 0) {
    SpecGroup& spec = *spec_->groups[g];
    const bool idle = spec.spec_reader != nullptr;
    std::unique_lock<std::mutex> turn(spec.turn, std::defer_lock);
    if (!idle) {
      // A demand blocked on the turn goes first: the unlocking thread would otherwise retake it before the woken
      // demand. Unbounded, but by the one demand read the flag stands for, which the watchdog times.
      while (spec.demand_waiting.load(std::memory_order_seq_cst) != 0)
        _mm_pause();
      turn.lock();
    }
    if (spec_stale(spec, source, target, source_seq) ||
        __atomic_load_n(map_ + target * experts_ + expert, __ATOMIC_ACQUIRE) >= 0 ||
        pool_->find(target, g, expert) >= 0) {
      spec_count<kSpecDropped>(spec);
      return;
    }
    int i = -1;
    {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      i = pool_->claimable_locked(target, g, source_seq);  // never a swapped entry: its victim may still be mapped
      if (i < 0) {  // every entry reading, swapped, or this job's own pick
        spec_count<kSpecDropped>(spec);
        return;
      }
      PoolEntry& entry = pool_->entry(target, g, i);
      entry.for_seq.store(source_seq, std::memory_order_relaxed);
      entry.boost.store(0, std::memory_order_relaxed);  // published by the release below
      entry.word.store(pool_word(kPoolReading, expert), std::memory_order_release);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);  // pairs with note_serving's
    if (spec_stale(spec, source, target, source_seq)) {
      // A record of the row went into service meanwhile; a forced miss that saw the claim waits for this store.
      // A landed entry reclaimed by the claim is lost with it, which only wastes its read.
      pool_->entry(target, g, i).word.store(kPoolEmpty, std::memory_order_release);
      spec_count<kSpecDropped>(spec);
      return;
    }
    spec_count<kSpecIssued>(spec);
    [[maybe_unused]] int64_t slot = 0;  // set and read by the InstrBuild events only
    if constexpr (Build::kMetrics) {
      slot = pool_->entry(target, g, i).slot;  // stable until a swap, which takes a landed entry
      // row = target row, seq = source record seq, a = expert, b = slot, c = source row; spec_use's c joins on seq.
      spec.trace.emit("spec_submit", target, pick, source_seq, g, expert, slot, source);
    }
    spec.busy.store(++spec.episodes, std::memory_order_release);
    bool fail = false;
    SpecPieces hooks{this, g, source, target, source_seq, 0};
    if constexpr (Build::kFaults) {
      Source& reader = pool_reader(g);
      const bool installed = apply_pending_fault(reader);  // reaches the reader this read uses, as for a demand
      if (const int64_t ns = faults_.spec_delay_ns.load()) {
        if (idle)
          hooks.delay_ns = ns;
        else
          fault_delay(ns);
      }
      fail = faults_.spec_fail.load();
      // The failed read replaces the one that would have met the fault: spend it, not leave it for the next demand.
      if (fail && installed) reader.set_fault(ReadFault{});
    }
    int result = 0;
    IdleRoots roots;
    if (fail) {
      pool_->entry(target, g, i).word.store(kPoolEmpty, std::memory_order_release);
    } else if (idle) {
      roots.boost = &pool_->entry(target, g, i).boost;
      roots.deadline_ns = spec_->config.idle_deadline_ns;
      roots.spread = spec_->config.idle_spread;
      roots.max_inflight = spec_->config.idle_pieces;
      roots.give_up = &SpecPieces::give_up;
      roots.on_piece = &SpecPieces::on_piece;
      roots.context = &hooks;
      result = land_pool_row(g, target, i, expert, &spec.packed, roots);
      spec_count<kSpecDeferred>(spec, roots.deferrals);
      spec_count<kSpecMovedRoot>(spec, roots.moves);
      if (roots.boosted) spec_count<kSpecBoosted>(spec);
    } else {
      result = land_pool_row(g, target, i, expert, &spec.packed);
    }
    spec.busy.store(0, std::memory_order_release);
    if (result == 1) {
      spec_count<kSpecLanded>(spec);
      if constexpr (Build::kMetrics) spec.trace.emit("spec_land", target, 0, source_seq, g, expert, slot, source);
    } else if (roots.abandoned) {
      spec_count<kSpecAbandoned>(spec);
    } else {
      spec_count<kSpecFailed>(spec);
    }
  }

  // An idle-drive read's hooks (IdleRoots::give_up, on_piece), on the speculative thread.
  struct SpecPieces {
    RamTier* tier;
    int g;
    int64_t source, target;
    uint32_t source_seq;
    int64_t delay_ns;  // inject_spec: the first piece in flight sleeps this long (InstrBuild)

    // Abandon before the next piece once the group has finished a record that made the read stale (a forced miss of
    // that record would have boosted it, and a boosted read never asks), or the threads are held or stopped.
    static bool give_up(void* context) {
      const auto& self = *static_cast<const SpecPieces*>(context);
      const SpecState& spec = *self.tier->spec_;
      return spec.hold.load(std::memory_order_relaxed) || spec.stop.load(std::memory_order_relaxed) ||
             self.tier->spec_settled(self.g, self.source, self.target, self.source_seq);
    }
    static void on_piece(void* context, int /*root*/) {
      auto& self = *static_cast<SpecPieces*>(context);
      if (self.delay_ns > 0) {
        self.tier->fault_delay(self.delay_ns);
        self.delay_ns = 0;
      }
    }
  };

  // True once group g has finished a record of `source` or `target` after `source_seq` (spec_stale's records): the read
  // landed too late for it, and no forced miss of it waits on the read.
  bool spec_settled(int g, int64_t source, int64_t target, uint32_t source_seq) const {
    const SpecGroup& spec = *spec_->groups[g];
    const uint32_t next = skip_zero(source_seq + 1u);
    const uint32_t handled = dist_.group(g).handled.load(std::memory_order_acquire);
    for (const int64_t row : {source, target}) {
      const uint32_t last = spec.row_seq[row].load(std::memory_order_acquire);
      if (last != 0 && reached(last, next) && reached(handled, last)) return true;
    }
    return false;
  }

  // The reader group g's pool rows are read through: its idle-drive reader, else the group's own.
  Source& pool_reader(int g) {
    if (spec_ != nullptr && spec_->groups[g]->spec_reader != nullptr) return *spec_->groups[g]->spec_reader;
    return dist_.group(g).reader;
  }

  // Names group g's speculative thread and pins it to its cores; returns 0 or the errno.
  int pin_spec(int g) {
    SpecGroup& spec = *spec_->groups[g];
    const std::string name = std::string(Layout::kName) + "-spec" + std::to_string(g);
    pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
    if (spec.cores.empty()) return 0;
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    for (const int core : spec.cores)
      CPU_SET(core, &cpus);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
  }

  // Group g's speculative thread: serves its ring, sleeping on its doorbell when the ring is empty.
  void run_spec(int g) {
    SpecGroup& spec = *spec_->groups[g];
    while (spec_wait_unheld(spec)) {
      SpecJob job;
      if (spec.ring.pop(&job)) {
        serve_spec_job(g, job, /*threaded=*/true);
        continue;
      }
      spec.idle.store(true, std::memory_order_seq_cst);
      spec.bell.sleep_unless([&] {
        return !spec.ring.empty() || spec_->hold.load(std::memory_order_relaxed) ||
               spec_->stop.load(std::memory_order_relaxed);
      });
    }
    spec.idle.store(true, std::memory_order_seq_cst);
  }

  // A speculative thread before a job or a read: marks it busy, or waits marked idle while quiesced; false once
  // stopped. `idle` is stored before `hold` is loaded, both seq_cst, so a quiescer that saw `idle` knows none starts.
  bool spec_wait_unheld(SpecGroup& spec) {
    while (!spec_->stop.load(std::memory_order_acquire)) {
      spec.idle.store(false, std::memory_order_seq_cst);
      if (!spec_->hold.load(std::memory_order_seq_cst)) return true;
      spec.idle.store(true, std::memory_order_seq_cst);
      // resume_spec and stop_spec ring the doorbell after their stores.
      spec.bell.sleep_unless([&] {
        return !spec_->hold.load(std::memory_order_relaxed) || spec_->stop.load(std::memory_order_relaxed);
      });
    }
    return false;
  }

  // Reads `expert`'s row of `row` into group g's pool entry i, which the caller claimed (kPoolReading), through
  // pool_reader(g) under root policy `roots`, then lands the entry, or empties it when the read failed or was
  // abandoned. Returns the read's result, 1 when the row landed.
  template <class Roots = TableRoots>
  int land_pool_row(int g, int64_t row, int i, int32_t expert, std::vector<uint8_t>* packed, Roots&& roots = Roots{}) {
    PoolEntry& entry = pool_->entry(row, g, i);
    const int64_t slot = entry.slot;  // stable: only a swap changes it, and a swap takes a landed entry
    int result = 0;
    Source& reader = pool_reader(g);
    // The drive load counts this read as speculative; the group's reader goes back to demand after it (an idle-drive
    // reader is speculative throughout).
    const bool shared = &reader == &dist_.group(g).reader;
    if (shared) reader.set_read_kind(kSpecRead);
    try {
      result = reader.read(
          row, std::span<const int32_t>(&expert, 1), std::span<const int64_t>(&slot, 1), 1,
          [](size_t) { return false; }, nullptr, packed, SIZE_MAX, NoProgress{}, nullptr, roots);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "ERROR %spool read: %s\n", error_prefix<Layout>().c_str(), error.what());
    }
    if (shared) reader.set_read_kind(kDemandRead);
    _mm_sfence();  // the row's bytes before the landed word
    if (result == 1) {
      entry.landed.store(pool_->next_landing(), std::memory_order_relaxed);
      entry.word.store(pool_word(kPoolLanded, expert), std::memory_order_release);
    } else {
      entry.word.store(kPoolEmpty, std::memory_order_release);
    }
    return result;
  }

  // The row's publisher, right after it publishes `request`'s delta: every group's entry swapped by this record becomes
  // empty, so no speculative read lands in a victim the device may still map. A reporter's swaps precede its report,
  // which the publisher's report acquires, so each is in place by now.
  void release_swapped(int64_t row, uint64_t chain) {
    for (int g = 0; g < dist_.size(); ++g) {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      for (int i = 0; i < pool_->share(); ++i) {
        PoolEntry& entry = pool_->entry(row, g, i);
        if (pool_state(entry.word.load(std::memory_order_relaxed)) == kPoolSwapped && entry.freed_at == chain)
          entry.word.store(kPoolEmpty, std::memory_order_release);
      }
    }
  }

  // Writes row's pending delta (claimed by the caller's CAS), then frees its swapped victims. Returns the promoted
  // count, or -1 when another thread already claimed it. `by` is the record whose serve triggered it; trigger is
  // 0 immediate, 1 claim from row-1's record, 2 backstop, 3 flush.
  int publish_pending(Group& group, const Request& by, int64_t row, int trigger) {
    uint64_t chain = 0;
    // The backstop takes only a delta stashed in an earlier forward than the caller's.
    const uint64_t forward = forward_[group.index].id;
    if (!dist_.claim(row, &chain, trigger == 2 ? forward : ~0ull)) {
      // Another group's serve of this record may have claimed it: its leftovers are not no_delta.
      if (trigger == 1 && pool_ != nullptr && !dist_.delta_taken(row, forward)) count_landed_no_delta(group, row);
      return -1;
    }
    if (trigger == 1) dist_.mark_claimed(row, forward);
    const int promoted = promote_landed(group, row, chain);
    dist_.write_pending(lease_, row);
    if (pool_ != nullptr) release_swapped(row, chain);
    if constexpr (Build::kMetrics)
      group.trace.emit(
          "delta_publish", row, by.gen, by.seq, group.index, static_cast<int64_t>(chain), static_cast<int64_t>(trigger),
          static_cast<int64_t>(promoted));
    return promoted;
  }

  // Trigger 1 found no delta waiting for `row`: its landed pool rows stay unpromoted, one count each. Every group serves
  // the record, so each counts only its own pool entries.
  void count_landed_no_delta(Group& claimer, int64_t row) {
    const int g = claimer.index;
    std::lock_guard<std::mutex> lock(pool_->mutex(g));
    for (int i = 0; i < pool_->share(); ++i)
      if (pool_state(pool_->entry(row, g, i).word.load(std::memory_order_relaxed)) == kPoolLanded)
        count<kSpecPromoteNoDelta>(claimer);
  }

  // Maps landed pool rows of `row` into its claimed delta (evicting a victim each), counts into the claimer's block,
  // and returns how many. The service thread takes row_mutex[row], then the pool mutex; the claimer takes the pool
  // mutex, then try_locks row_mutex[row], and when busy leaves the entries landed (spec_promote_row_busy each).
  int promote_landed(Group& claimer, int64_t row, uint64_t chain) {
    if (pool_ == nullptr) return 0;
    int promoted = 0;
    Tier& tier = tiers_[row];
    for (int g = 0; g < dist_.size(); ++g) {
      Group& owner = dist_.group(g);
      std::lock_guard<std::mutex> pool_lock(pool_->mutex(g));
      std::unique_lock<std::mutex> row_lock(owner.row_mutex[row], std::try_to_lock);
      if (!row_lock.owns_lock()) {
        for (int i = 0; i < pool_->share(); ++i)
          if (pool_state(pool_->entry(row, g, i).word.load(std::memory_order_acquire)) == kPoolLanded)
            count<kSpecPromoteRowBusy>(claimer);
        continue;
      }
      GroupRow& own = owner.rows[row];
      for (int i = 0; i < pool_->share(); ++i) {
        PoolEntry& entry = pool_->entry(row, g, i);
        const uint32_t word = entry.word.load(std::memory_order_acquire);
        if (pool_state(word) == kPoolReading) {
          count<kSpecPromoteReading>(claimer);
          continue;
        }
        if (pool_state(word) != kPoolLanded) continue;
        const int32_t expert = pool_expert(word);
        if (tier.expert_slot[expert] >= 0) continue;  // already mapped: an ordinary hit
        if (dist_.pending_count(row) + 2 > Wire::kDeltaMaxEntries) {
          count<kSpecPromoteNoRoom>(claimer);
          continue;
        }
        int32_t old = -1;
        const int64_t victim = take_victim_locked(owner, row, {}, &old, false);
        if (victim < 0) {
          count<kSpecPromoteNoVictim>(claimer);
          continue;
        }
        const int32_t slot = entry.slot;
        if (old >= 0) dist_.append_pending(row, old, -1);
        dist_.append_pending(row, expert, slot);
        tier.expert_slot[expert] = slot;
        tier.slot_to_expert[slot] = expert;
        tier.state[slot] = kReady;
        tier.stamp[slot] = tier.stamp[victim];  // the promoted row inherits the victim's recency
        publish_mirror(row, expert, slot);
        tier.state[victim] = kSpec;
        entry.slot = static_cast<int32_t>(victim);
        entry.freed_at = chain;
        entry.word.store(pool_word(kPoolSwapped, kPoolNoExpert), std::memory_order_release);
        if (own.promoted_chain != chain) {
          own.promoted.clear();
          own.promote_evicted.clear();
          own.promoted_chain = chain;
        }
        own.promoted.push_back(expert);
        own.promote_evicted.push_back(old);
        count<kSpecPromotedEarly>(claimer);
        ++promoted;
      }
    }
    return promoted;
  }

  // Deferred publication for one served or skipped record: the backstop for the record's own row and for earlier
  // rows stranded since an earlier forward, then the claim of row + 1's delta, which this record's row precedes.
  void publish_due(Group& group, const Request& request) {
    if (request.row < 0 || request.row >= layers_) return;
    ForwardState& fwd = forward_[group.index];
    if (request.row <= fwd.last_row) ++fwd.id;
    fwd.last_row = request.row;
    // A delta another group stashed from this very record (same forward) still waits for row - 1's next serve.
    for (int64_t p = 0; p <= request.row; ++p)
      if (dist_.deferred(p) && publish_pending(group, request, p, 2) >= 0)
        count<kSpecPublishBackstop>(group);
    if (request.row + 1 < layers_ && dist_.deferred(request.row + 1))
      publish_pending(group, request, request.row + 1, 1);
  }

  // Picks the victims and publishes the delta, before any read, so a served chain always has its delta published. A
  // read miss lands in its staging slot whatever happens here (a pool-served one is streamed from its pool slot);
  // whether it is cached is the victim's question. inserted[i] is set when miss i took a victim and is cached once read.
  void reserve_victims_locked(
      Group& group, Tier& tier, const Request& request, RecordPlan& plan, bool* inserted) {
    GroupRow& own = group.rows[request.row];
    DeltaReport part;
    part.staging = own.staging;
    // Forced CPU misses first (spill, slot -1): each is read straight into a victim, which it takes over, or swapped in
    // from the pool, so it is cached with no staging slot. The start-up capacity check (Exl3RamMissService.attach)
    // leaves every node range a victim for each, so none missing is a broken invariant, not a skipped insert.
    bool placed[Wire::kLanes] = {};
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      if (plan.slots[i] >= 0) continue;
      int32_t old = -1;
      const int64_t victim = take_victim_locked(group, request.row, plan.wanted, &old);
      if (victim < 0) fail_record(request, "a forced CPU miss found no RAM victim on its node");
      if constexpr (Build::kMetrics) trace_evict(group, tier, request, victim, old, plan.missing[i]);
      if (old >= 0) {
        part.entries[part.count][0] = old;
        part.entries[part.count][1] = -1;
        ++part.count;
      }
      // A pooled row is mapped where it landed, and the victim takes its place in the pool once the delta publishes.
      const int32_t pooled =
          pool_ != nullptr ? take_pooled_locked(group, request, plan.missing[i], static_cast<int32_t>(victim)) : -1;
      const int64_t slot = pooled >= 0 ? pooled : victim;
      part.entries[part.count][0] = plan.missing[i];
      part.entries[part.count][1] = static_cast<int32_t>(slot);
      ++part.count;
      tier.state[slot] = kStaging;  // being read, or landed in the pool; commit_inserted_locked makes it READY
      if (pooled >= 0) {
        tier.state[victim] = kSpec;
        plan.pooled |= Wire::LaneMask{1} << i;
      }
      plan.slots[i] = slot;
      placed[i] = true;
      inserted[i] = true;
    }
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      if (placed[i]) continue;
      int32_t old = -1;
      const int64_t victim = take_victim_locked(group, request.row, plan.wanted, &old);
      const Lane& lane = request.lanes[plan.miss_lane[i]];
      const int32_t pooled =
          pool_serve_ && lane.kind == Wire::kKindMissGpu ? point_pooled_locked(group, request, plan, i, victim) : -1;
      if (pooled >= 0) {
        // The staging slot stays staging: S streams from the pool slot instead (the lane's PieceSource word).
        plan.slots[i] = pooled;
        plan.pooled |= Wire::LaneMask{1} << i;
      } else if constexpr (Build::kMetrics) {
        if (pool_ != nullptr) plan.pool_seen[i] = count_skipped_pooled(group, request.row, lane);
      }
      if (victim < 0) {
        this->template count<kRamInsertSkipped>(group);
        continue;
      }
      if constexpr (Build::kMetrics) trace_evict(group, tier, request, victim, old, plan.missing[i]);
      if (old >= 0) {
        part.entries[part.count][0] = old;
        part.entries[part.count][1] = -1;
        ++part.count;
      }
      part.entries[part.count][0] = plan.missing[i];
      part.entries[part.count][1] = static_cast<int32_t>(plan.slots[i]);
      ++part.count;
      const int32_t slot = static_cast<int32_t>(plan.slots[i]);
      if (pooled >= 0) {
        // The pool slot is READY once commit_inserted_locked runs, while S may still stream it: safe as any hit's
        // slot, since no later record of the row posts before this chain (and S) ended.
        tier.state[slot] = kStaging;
        tier.state[victim] = kSpec;
      } else {
        tier.state[victim] = kStaging;
        for (int32_t& s : part.staging)
          if (s == slot) s = static_cast<int32_t>(victim);
      }
      inserted[i] = true;
    }
    own.staging = part.staging;
    const uint32_t reporters = NumaNodeDistributor<Source>::miss_nodes(request);
    if (dist_.report(request, group.index, part, reporters)) {
      if (dist_.deferred(request.row)) {
        // Written by the serve of row - 1's next record; its swapped victims are released then.
        if (!dist_.stash(request, reporters, forward_[group.index].id))
          fail_record(request, "the row's previous map delta is not chain " + std::to_string(request.chain - 1));
      } else {
        if (!dist_.publish(lease_, request, reporters))
          fail_record(request, "the row's previous map delta is not chain " + std::to_string(request.chain - 1));
        if (pool_ != nullptr) release_swapped(request.row, request.chain);
        if constexpr (Build::kMetrics) {
          if (promote_)
            group.trace.emit(
                "delta_publish", request.row, request.gen, request.seq, group.index,
                static_cast<int64_t>(request.chain), int64_t{0}, int64_t{0});
        }
      }
    }
  }

  // The rows the record's misses read: a miss swapped in from the pool reads none.
  static int64_t rows_read(const RecordPlan& plan) {
    return static_cast<int64_t>(plan.missing.size()) - std::popcount(plan.pooled);
  }

  // True once miss i's row is in RAM: swapped in from the pool, or packed by the read (indexed by its read ordinal when
  // some miss was pooled).
  static bool miss_landed(const Group& group, const RecordPlan& plan, size_t i) {
    if ((plan.pooled >> i & 1u) != 0) return true;
    const size_t at = plan.pooled != 0 ? static_cast<size_t>(plan.ordinal[i]) : i;
    return at < group.packed.size() && group.packed[at] != 0;
  }

  // True once miss i's first stage is in RAM (two-stage CPU misses): its whole row, or its read's first span.
  static bool miss_first_stage_landed(const Group& group, const RecordPlan& plan, size_t i) {
    if (miss_landed(group, plan, i)) return true;
    const size_t at = plan.pooled != 0 ? static_cast<size_t>(plan.ordinal[i]) : i;
    return at < group.prefix.size() && group.prefix[at] != 0;
  }

  // Opens the second stage of each staged job whose rows have all landed whole.
  void open_landed_stage_two(Group& group, const RecordPlan& plan, CpuMissBatch* misses) {
    for (int j = 0; j < misses->staged;) {
      const Wire::LaneMask rows = misses->staged_rows[j];
      bool whole = true;
      for (size_t i = 0; i < plan.missing.size() && whole; ++i)
        whole = (rows >> i & 1u) == 0 || miss_landed(group, plan, i);
      if (!whole) {
        ++j;
        continue;
      }
      _mm_sfence();  // the rows' bytes before the CPU thread reads them
      group.cpu->open_stage_two(misses->staged_seq[j]);
      --misses->staged;
      misses->staged_seq[j] = misses->staged_seq[misses->staged];
      misses->staged_rows[j] = misses->staged_rows[misses->staged];
    }
  }

  // Submits the CPU misses whose rows landed since the last call as one part-1 job. Called from read()'s progress hook
  // and after the read. Every job after the record's first adds into the part. With two-stage CPU misses a miss goes
  // once its first stage landed, and a job holding a row not yet whole is staged: its second stage opens once they are.
  void submit_landed_cpu_misses(Group& group, const Request& request, const RecordPlan& plan, CpuMissBatch* misses) {
    const SvcTimer timer(svc_[group.index].cpu_ns);
    if (misses->staged != 0) open_landed_stage_two(group, plan, misses);
    if (misses->left == 0) return;
    const CopyJob& job = plan.job;
    const bool two_stage = group.cpu->config().two_stage;
    CpuJob cpu_job;
    cpu_job.row = request.row;
    cpu_job.part = 1;
    cpu_job.accumulate = misses->left < job.late_cpu;
    cpu_job.per_token = true;
    Wire::LaneMask landing = 0;  // the job's misses not yet whole
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      const Lane& lane = request.lanes[plan.miss_lane[i]];
      if (lane.kind != Wire::kKindMissCpu || (misses->sent >> i & 1u) != 0) continue;
      if (!miss_landed(group, plan, i)) {
        if (!two_stage || !miss_first_stage_landed(group, plan, i)) continue;
        landing |= Wire::LaneMask{1} << i;
      }
      cpu_job.slots[cpu_job.k] = static_cast<int32_t>(plan.slots[i]);
      cpu_job.weights[cpu_job.k] = lane.weight;
      cpu_job.lanes[cpu_job.k] = static_cast<int32_t>(plan.miss_lane[i]);
      ++cpu_job.k;
      misses->sent |= Wire::LaneMask{1} << i;
    }
    if (cpu_job.k == 0) return;
    misses->left -= cpu_job.k;
    cpu_job.seq = misses->left == 0 ? job.late_seq : misses->next++;
    if (landing != 0) {
      cpu_job.staged = true;
      misses->staged_seq[misses->staged] = cpu_job.seq;
      misses->staged_rows[misses->staged] = landing;
      ++misses->staged;
    }
    _mm_sfence();  // the rows' bytes before the CPU thread reads them
    submit_cpu_job(group, request, cpu_job);
  }

  // Reads the misses into their staging slots, each CPU miss going to the CPU as its row lands. A miss swapped in from
  // the speculative pool is not read and its CPU job goes first. Fail-stops on a failed read. Returns the stage status.
  int64_t read_misses(
      Group& group, const Request& request, RecordPlan& plan, CpuMissBatch* misses, StageRecord* cur) {
    group.packed.clear();
    group.prefix.clear();
    const bool pooled = plan.pooled != 0;
    if (pooled) {
      for (size_t i = 0; i < plan.missing.size(); ++i) {
        if ((plan.pooled >> i & 1u) != 0) {
          plan.ordinal.push_back(-1);
          continue;
        }
        plan.ordinal.push_back(static_cast<int32_t>(plan.read_experts.size()));
        plan.read_experts.push_back(plan.missing[i]);
        plan.read_slots.push_back(plan.slots[i]);
        plan.read_lanes.push_back(plan.miss_lane[i]);
      }
      submit_landed_cpu_misses(group, request, plan, misses);
    }
    const std::span<const int32_t> experts = pooled ? plan.read_experts.span() : plan.missing.span();
    const std::span<const int64_t> slots = pooled ? plan.read_slots.span() : plan.slots.span();
    const std::span<const int32_t> lanes = pooled ? plan.read_lanes.span() : plan.miss_lane.span();
    if (!experts.empty()) {
      read_rows(group, request, plan, experts, slots, lanes, misses, cur);
      if constexpr (Build::kMetrics) {
        if (cur) trace_miss_lands(group, request, plan, cur);
      }
    } else {
      // Every miss came from the pool: a read of no rows. It counts as a demand, and its read stamps sit at the
      // reservation (no clock read), so the stage order holds and the record shows no read time.
      ++group.demands_read;
      if (cur) cur->submit = cur->first_cqe = cur->last_cqe = cur->pack_start = cur->pack_end = cur->reserved;
    }
    submit_landed_cpu_misses(group, request, plan, misses);
    if (misses->left != 0) fail_record(request, "a CPU miss's row never landed");
    if (misses->staged != 0) fail_record(request, "a staged CPU miss's row never landed whole");
    return kStatusServed;
  }

  // One miss_land per read miss, from the stage record (so only with the stage trace on): a = expert, b = when its row
  // landed (its last piece vetted, else its packing ended; 0 past kTraceRows), c = its PoolSeen | 4 for a CPU miss.
  void trace_miss_lands(Group& group, const Request& request, const RecordPlan& plan, const StageRecord* cur)
    requires(Build::kMetrics)
  {
    const bool pooled = plan.pooled != 0;
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      const int32_t o = pooled ? plan.ordinal[i] : static_cast<int32_t>(i);
      if (o < 0) continue;
      int64_t landed = 0;
      if (o < kTraceRows) {
        for (int j = 0; j < kPieces; ++j)
          landed = std::max(landed, cur->piece_cqe[o][j]);
        if (landed == 0) landed = cur->row_pack_end[o];
      }
      const bool cpu = request.lanes[plan.miss_lane[i]].kind == Wire::kKindMissCpu;
      group.trace.emit("miss_land", request.row, request.gen, request.seq, group.index, plan.missing[i], landed,
                       static_cast<int64_t>(plan.pool_seen[i]) | (cpu ? 4 : 0));
    }
  }

  // read_misses' demand read of `experts` into `slots` (`lanes` their record lanes), in read order.
  void read_rows(
      Group& group,
      const Request& request,
      RecordPlan& plan,
      std::span<const int32_t> experts,
      std::span<const int64_t> slots,
      std::span<const int32_t> lanes,
      CpuMissBatch* misses,
      StageRecord* cur) {
    const SvcTimer timer(svc_[group.index].read_ns);
    // The group's reader takes turns with its speculative thread: a demand waits for at most the one speculative row.
    // An idle-drive speculative thread reads through its own reader and takes no turn.
    std::unique_lock<std::mutex> turn;
    if (spec_ != nullptr && spec_->groups[group.index]->spec_reader == nullptr) {
      turn = std::unique_lock<std::mutex>(spec_->groups[group.index]->turn, std::try_to_lock);
      if (!turn.owns_lock()) {
        count<kSpecDelayed>(group);
        std::atomic<int>& waiting = spec_->groups[group.index]->demand_waiting;
        waiting.fetch_add(1, std::memory_order_seq_cst);  // spec_read yields the turn to a waiting demand
        struct Lower {  // a throwing lock must not leave the count raised
          std::atomic<int>& count;
          ~Lower() { count.fetch_sub(1, std::memory_order_relaxed); }
        } lower{waiting};
        turn.lock();
      }
    }
    const bool publishing = init_piece_words_locked(group, request, lanes, plan.idx);
    // Two-stage CPU misses: each CPU miss's row in two spans, its first stage over every root before its second.
    const bool two_stage = group.cpu != nullptr && group.cpu->config().two_stage;
    group.two_span.clear();
    for (size_t o = 0; two_stage && o < lanes.size(); ++o)
      group.two_span.push_back(request.lanes[lanes[o]].kind == Wire::kKindMissCpu ? 1 : 0);
    bool fail_reads = false;
    if constexpr (Build::kFaults) {  // the test faults (inject, inject_fault): InstrBuild only
      apply_pending_fault(group.reader);
      const int64_t delay = faults_.delay_ns.load();
      if (delay > 0 && group.demands_read >= faults_.delay_after.load()) fault_delay(delay);
      fail_reads = faults_.fail_reads.load();
    }
    if (fail_reads) fail_record(request, "a test fault failed the read");
    const int result = group.reader.read(
        request.row,
        experts,
        slots,
        kBounceRows,
        [](size_t) { return false; },
        cur,
        &group.packed,
        SIZE_MAX,
        // read() runs this once per drain-loop turn and once per finished row.
        [&]() { submit_landed_cpu_misses(group, request, plan, misses); },
        publishing ? &group.piece_publish : nullptr,
        TableRoots{},
        TwoSpanRows{two_stage ? group.two_span.data() : nullptr, two_stage ? &group.prefix : nullptr});
    // The tier's count is the sum over the groups' readers: each adds what its own reader refused since it last did.
    const int64_t refused = group.reader.publish_refused();
    stats_.add(kPiecePublishRefused, refused - group.publish_refused_seen);
    group.publish_refused_seen = refused;
    if (result != 1) {
      count<kReadErrors>(group);
      fail_record(request, "the read failed");
    }
    ++group.demands_read;
    _mm_sfence();  // the pieces' memcpy stores land before the mirror publishes them
  }

  // Maps each miss that took a victim (read, or swapped in from the pool) into the tier and the host mirror.
  void commit_inserted_locked(
      Group& group, Tier& tier, const Request& request, const RecordPlan& plan, const bool* inserted) {
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      if (!inserted[i]) continue;
      tier.slot_to_expert[plan.slots[i]] = plan.missing[i];
      tier.expert_slot[plan.missing[i]] = static_cast<int32_t>(plan.slots[i]);
      tier.state[plan.slots[i]] = kReady;
      tier.stamp[plan.slots[i]] = ++group.tick;
      publish_mirror(request.row, plan.missing[i], static_cast<int32_t>(plan.slots[i]));
    }
    // Any group adds, layer_rows reads lock-free: relaxed.
    const int64_t read = rows_read(plan);
    std::atomic_ref<int64_t>(tier.rows_demand).fetch_add(read, std::memory_order_relaxed);
    count<kVersion>(group);
    count<kRowsRead>(group, read);
  }

  // Serves a record with lanes (analysis/dsv41-drive/LEASE_PROTOCOL.md, "The host per record"). The device typed every
  // lane from its copy of the map, so the host checks each against the tier and fail-stops on any disagreement.
  //
  // Order: stamps, checks, the copy job (hits start at once), victims and the map delta (before any read, so a served
  // chain always has its delta published), then the misses' reads. Returns whether any miss was placed: read, or
  // swapped in from the pool.
  bool serve_record(Group& group, const Request& request) {
    StageRecord* const cur = stage_record(group);  // null in ProdBuild, so every `if (cur)` below folds away
    if (cur) cur->lanes = static_cast<int64_t>(request.lanes.size());
    if (request.row < 0 || request.row >= layers_) fail_record(request, "the row is out of range");
    Tier& tier = tiers_[request.row];
    RecordPlan plan;
    plan.idx = static_cast<int64_t>((request.seq - 1u) % Wire::kDemandRecords);
    collect_wanted_locked(group, tier, request, &plan);
    classify_lanes_locked(group, tier, request, &plan);
    CpuMissBatch misses = submit_host_lanes(group, request, &plan);
    const bool reads = !plan.missing.empty();
    bool inserted[Wire::kLanes] = {};
    if (reads) reserve_victims_locked(group, tier, request, plan, inserted);
    if (cur) cur->reserved = stamp(cur);
    const int64_t status = reads ? read_misses(group, request, plan, &misses, cur) : kStatusNoRead;
    if (reads) commit_inserted_locked(group, tier, request, plan, inserted);
    const int64_t read = rows_read(plan);
    if (cur) {
      cur->mapped = stamp(cur);
      cur->row = request.row;
      cur->ok = 1;
      cur->status = status;
      cur->rows = read;
    }
    return reads;
  }

  // Stores piece_word(gen) into the readiness word of every kMissGpu lane, fences, and records those words as the
  // reader's publish targets, by the row's ordinal in the read. A kMissCpu lane has no device reader, so no word.
  // Returns false when no lane has one.
  bool init_piece_words_locked(Group& group, const Request& request, std::span<const int32_t> miss_lane, int64_t idx) {
    group.piece_targets.assign(miss_lane.size(), PieceTarget{});
    bool any = false;
    for (size_t i = 0; i < miss_lane.size(); ++i) {
      const int32_t lane = miss_lane[i];
      if (request.lanes[lane].kind != Wire::kKindMissGpu) continue;
      uint8_t* word =
          lease_ + Wire::kLeasePieceMask +
          (idx * Wire::kLanes + static_cast<int64_t>(lane)) * Wire::kLeasePieceMaskLineBytes;
      store_release64(word, piece_word(request.gen));
      PieceTarget& target = group.piece_targets[i];
      target.words[target.count++] = reinterpret_cast<uint64_t*>(word);
      any = true;
    }
    _mm_sfence();
    group.piece_publish = PiecePublish{request.gen, group.piece_targets.data()};
    return any;
  }

  // Serves one record: a lane-less one only refreshes recency, one with lanes goes to serve_record. Held in one busy
  // episode so the watchdog can time it.
  void handle_record(Group& group, const Request& request) {
    int64_t svc_start = 0;
    if constexpr (Build::kMetrics) {
      svc_start = now_ns();
      SvcClock& svc = svc_[group.index];
      svc.cpu_ns = svc.read_ns = 0;
    }
    begin_busy(group);
    // A promoter of another group may map into this row's tier; it takes the same lock.
    std::unique_lock<std::mutex> row_lock;
    if (promote_ && request.row >= 0 && request.row < layers_)
      row_lock = std::unique_lock<std::mutex>(group.row_mutex[request.row]);
    if (!request.lanes.empty()) {
      if (serve_record(group, request)) {
        count<kServedRequests>(group);
      } else {
        count<kTouchOnly>(group);
      }
    } else {
      touch_request(group, request);
      count<kTouchOnly>(group);
      if (StageRecord* const cur = stage_record(group)) {
        cur->kind = kStageTouch;
        cur->row = request.row;
        cur->ok = 1;
        cur->status = kStatusTouch;
      }
    }
    end_busy(group);
    if constexpr (Build::kMetrics) trace_svc(group, request, svc_start);
  }

  // svc_record: a = the record's service ns, b = the idle ns since the group's previous record ended (-1 for its
  // first), c = ns inside read_rows (which includes the side work its drain loop runs). svc_side: a = ns submitting
  // CPU jobs, also inside the read when its loop ran them; b = 0, since no pool row is copied. svc_routed: a = the
  // routed experts the group homes, b = all routed experts, c = all lanes; a routed expert with no lane is VRAM-hot.
  void trace_svc(Group& group, const Request& request, int64_t start)
    requires(Build::kMetrics)
  {
    SvcClock& svc = svc_[group.index];
    const int64_t end = now_ns();
    const int64_t idle = svc.last_end > 0 ? start - svc.last_end : -1;
    group.trace.emit("svc_record", request.row, request.gen, request.seq, group.index, end - start, idle, svc.read_ns);
    group.trace.emit("svc_side", request.row, request.gen, request.seq, group.index, svc.cpu_ns, 0, 0);
    int64_t own = 0;
    for (int32_t expert : request.protect)
      own += Wire::home(expert) == group.index;
    group.trace.emit(
        "svc_routed", request.row, request.gen, request.seq, group.index, own,
        static_cast<int64_t>(request.protect.size()), static_cast<int64_t>(request.lanes.size()));
    svc.last_end = end;
  }
