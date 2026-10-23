// RamTier private members: tier-owner helpers, slot ownership and eviction, the record plan, and CPU-miss submission.
// A class-body fragment: ram_tier.h includes it inside RamTier, at the place these members always sat. Include
// ram_tier.h, never this file.
#pragma once

  // One group's speculative state (enable_ram_prefetch): its job ring from the group's service thread, its reader (the
  // turn on the group's, or its own), its busy episode for the watchdog and its own counter block (one writer each).
  struct SpecGroup {
    explicit SpecGroup(const std::string& name) : trace(name) {}
    SpscRing<SpecJob, 64> ring;  // arbitrary: one job per CPU record, the thread drains it every ~layer
    Doorbell bell;
    // The shared reader only: one demand read (read_rows) or one speculative read at a time on the group's reader.
    std::mutex turn;
    std::atomic<int> demand_waiting{0};  // demand reads blocked on `turn`, which spec_read lets go first
    // SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH_IDLE_DRIVE: the group's speculative reader, its own ring (null: the group's reader,
    // under `turn`). The speculative thread's alone, and the owner's while it is quiesced.
    std::unique_ptr<Source> spec_reader;
    std::promise<int> pinned;  // start_spec: 0 pinned, else the errno; a member, so it outlives the set_value
    std::atomic<uint64_t> busy{0};
    uint64_t episodes = 0;
    std::atomic<bool> idle{true};
    GateScorer scorer;
    std::vector<uint8_t> skip;
    std::vector<uint8_t> packed;
    std::vector<int> cores;
    // Per row: the seq of its last record the group's service handled, 0 before any. Service stores, scorer loads.
    std::unique_ptr<std::atomic<uint32_t>[]> row_seq;
    std::thread thread;
    LineCounters<kCounterCount> core;
    [[no_unique_address]] JobTrace<Build::kMetrics> trace;
  };
  struct SpecState {
    RamPrefetchConfig config;
    const uint8_t* x_base = nullptr;  // the CPU rows' staged inputs (CpuExpertConfig)
    int64_t x_stride = 0;
    int64_t x_token_bytes = 0;
    int64_t tokens_max = 1;
    std::vector<std::unique_ptr<SpecGroup>> groups;
    std::atomic<bool> stop{false};
    std::atomic<bool> hold{false};
  };

  // Test only (InstrBuild): inject()'s read delay, standing in for a slow read.
  void fault_delay(int64_t ns) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
  }

  // One group per node of the build, over its slot range of every row. Throws on a range count other than
  // Wire::kNodes, a range outside its row, or two that overlap.
  static std::vector<std::unique_ptr<Group>> make_groups(
      const Tables& tables,
      const std::vector<int64_t>& capacity,
      bool direct,
      const std::vector<std::vector<std::pair<int64_t, int64_t>>>& ranges,
      const std::vector<int>& sq_thread_cpus) {
    if (static_cast<int>(ranges.size()) != Wire::kNodes || static_cast<int>(sq_thread_cpus.size()) != Wire::kNodes)
      throw std::runtime_error(
          error_prefix<Layout>() + "this build serves " + std::to_string(Wire::kNodes) + " NUMA groups");
    std::vector<std::unique_ptr<Group>> groups;
    for (int g = 0; g < Wire::kNodes; ++g) {
      if (static_cast<int64_t>(ranges[g].size()) != tables.layers)
        throw std::runtime_error(error_prefix<Layout>() + "a NUMA group needs a slot range per row");
      std::vector<GroupRow> rows(static_cast<size_t>(tables.layers));
      for (int64_t row = 0; row < tables.layers; ++row) {
        const auto [lo, hi] = ranges[g][row];
        if (lo < 0 || hi > capacity[row] || lo > hi)
          throw std::runtime_error(
              error_prefix<Layout>() + "group " + std::to_string(g) + "'s slots of row " + std::to_string(row) +
              " are outside the row");
        for (int other = 0; other < g; ++other) {
          const auto [olo, ohi] = ranges[other][row];
          if (lo < ohi && olo < hi)
            throw std::runtime_error(
                error_prefix<Layout>() + "groups " + std::to_string(other) + " and " + std::to_string(g) +
                " overlap in row " + std::to_string(row));
        }
        rows[row].lo = lo;
        rows[row].hi = hi;
      }
      groups.push_back(std::make_unique<Group>(
          g, sq_thread_cpus[g], tables, direct, std::move(rows), std::string(Layout::kName) + "-tier" + std::to_string(g)));
    }
    return groups;
  }

  // Throws "<what> needs the service thread paused" unless the caller owns the tier.
  void require_owner(const char* what) const {
    if (!caller_owns()) throw std::runtime_error(error_prefix<Layout>() + what + " needs the service thread paused");
  }

  // Copy thread. Called once every DMA and CPU job of the record is done: publishes CopyDone, so the device's wait ends
  // as soon as the work did, and opens the gate if the copy wait closed it.
  void copy_completed(const CopyJob& job) {
    // Dekker with the copy wait kernel (the channel's close_gate in CW): job.idx is the ring index of G's seq.
    channel::complete<TargetChannel>(lease_, static_cast<uint32_t>(job.gen), job.gen);
  }

  // Copy thread, or the submitter when the copy engine's job ring is full. Completion cannot be established and the
  // device would wait on CopyDone until the watchdog fires, so fail stop.
  void copy_failed(const CopyJob& job, int error) {
    fail_stop(
        std::string(Layout::kName) + " RAM miss copy engine: copy of request " + std::to_string(job.gen) + " failed (" +
        std::to_string(error) + ")");
  }

  // Starts the stage record of the request just found. With the trace off this is one relaxed load and no clock read;
  // requests are served one at a time, so one member record serves them all. ProdBuild: nothing.
  void begin_stage(int64_t kind, uint32_t seq, uint32_t backlog) {
    if constexpr (Build::kMetrics) {
      if (!trace_.on.load(std::memory_order_relaxed)) {
        trace_.cur = nullptr;
        return;
      }
      StageRecord& stage = trace_.stage;
      const int64_t observed = stamp(&stage);  // before the reset below: the record is found, not built
      stage = StageRecord{};
      stage.observed = observed;
      stage.kind = kind;
      stage.seq = seq;
      stage.backlog = backlog;
      stage.prev_done = trace_.last_done;
      const Source& reader = dist_.group(0).reader;  // the traced group
      stage.pack_workers = reader.pack_workers();
      stage.pack_split = reader.pack_split();
      stage.piece_stream = reader.piece_stream() ? 1 : 0;
      trace_.cur = &stage;
    } else {
      (void)kind;
      (void)seq;
      (void)backlog;
    }
  }

  // Before a read: installs the fault inject_fault() left on `reader`, which only the calling thread drives (a group's
  // service or fill on its reader, a speculative thread on the reader it reads through). The first read takes it.
  // Returns true when it installed one. ProdBuild: nothing.
  bool apply_pending_fault(Source& reader) {
    if constexpr (Build::kFaults) {
      if (!faults_.fault_pending.load(std::memory_order_acquire)) return false;
      ReadFault fault;
      {
        std::lock_guard<std::mutex> guard(faults_.fault_mutex);
        if (!faults_.fault_pending.load(std::memory_order_relaxed)) return false;  // another group's reader took it
        fault = faults_.pending_fault;
        faults_.fault_pending.store(false, std::memory_order_relaxed);
      }
      reader.set_fault(fault);
      return true;
    }
    (void)reader;
    return false;
  }

  // Completes and pushes the stage record begun by begin_stage, if any.
  void end_stage() {
    if constexpr (Build::kMetrics) {
      StageRecord* cur = trace_.cur;
      if (cur == nullptr) return;
      cur->done = stamp(cur);
      trace_.last_done = cur->done;
      trace_.ring->push(*cur);
      trace_.cur = nullptr;
    }
  }

  // The stage record of the request in service, or null; always null in ProdBuild, and for every group but 0 (one
  // StageRecord, filled by one thread).
  StageRecord* stage_record(const Group& group) const {
    if constexpr (Build::kMetrics) {
      return group.index == 0 ? trace_.cur : nullptr;
    } else {
      (void)group;
      return nullptr;
    }
  }

  [[noreturn]] static void throw_no_trace() {
    throw std::runtime_error(
        error_prefix<Layout>() +
        "the stage trace is in the instrumented host build only "
        "(set SGLANG_EXPERT_STREAM_TRACE's expert_path so the service loads it)");
  }

  template <Counter K>
  void count_into(LineCounters<kCounterCount>& core, int64_t n) {
    if constexpr (is_core_counter(K)) {
      core.add(K, n);
    } else {
      stats_.add(K, n);
    }
  }

  // The busy episode brackets a demand or fill in service (see busy_episode()).
  void begin_busy(Group& group) {
    group.busy.store(++group.episodes, std::memory_order_release);
  }
  void end_busy(Group& group) {
    group.busy.store(0, std::memory_order_release);
  }

  // Rounds up to a 4096-byte page.
  static int64_t round_up_page(int64_t value) {
    return (value + 4095) / 4096 * 4096;
  }

  // The size of the lease block: the completion block, then one delta record per row
  // (expert_lease_block.lease_block_bytes).
  static int64_t lease_block_bytes(int64_t rows) {
    return Wire::kLeaseBlockBytes + round_up_page(rows * Wire::kDeltaStride);
  }

  // Validates the lease block and opens its gate. Runs before the service thread or any device exists, so plain stores
  // and one fence suffice.
  void init_lease_block(int64_t lease_bytes) {
    if (lease_ == nullptr || reinterpret_cast<uintptr_t>(lease_) % Wire::kLeaseBlockAlign != 0) {
      throw std::runtime_error(error_prefix<Layout>() + "the lease block must be a 4096-byte aligned block");
    }
    if (lease_bytes != lease_block_bytes(layers_)) {
      throw std::runtime_error(
          error_prefix<Layout>() + "the lease block has " + std::to_string(lease_bytes) + " bytes, not " +
          std::to_string(lease_block_bytes(layers_)) + " for " + std::to_string(layers_) + " rows");
    }
    // Open with nothing armed: a copy wait that closes nothing passes its stream wait on this value (0 would block).
    const uint32_t open = gate_word(0, Wire::kLeaseGateOpen);
    std::memcpy(lease_ + Wire::kLeaseCopyGate, &open, 4);
    _mm_sfence();
  }

  // The free and evictable counts behind victim_census; "evictable" follows take_victim_locked's exclusions.
  VictimCensus census_locked(int64_t row, std::span<const int32_t> wanted) const {
    const Tier& tier = tiers_[row];
    VictimCensus census;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kFree) {
        ++census.free;
      } else if (tier.state[slot] == kReady) {
        const int32_t expert = tier.slot_to_expert[slot];
        if (tier.hot[expert] || tier.filling[slot] || listed(wanted, expert)) continue;
        ++census.evictable;
      }
    }
    return census;
  }

  // The fill thread body. Reads the claimed rows and publishes progress through fill_landed_ and fill_state_; touches
  // no tier state (the owner's epilogue, finish_fill_owned, does).
  void run_fill() {
    // Every group is parked during a fill and every ring registers the whole tier, so the fill reads through group 0's
    // and holds its busy episode.
    Group& group = dist_.group(0);
    begin_busy(group);
    apply_pending_fault(group.reader);            // test only: inject_fault() acts on a fill's read as on a demand's
    std::vector<uint8_t>& packed = fill_packed_;  // reserved to the widest row at construction: no allocation here
    size_t landed = 0;
    auto advance = [&] {
      while (landed < packed.size() && packed[landed] != 0)
        ++landed;
      fill_landed_.store(static_cast<int64_t>(landed), std::memory_order_release);
    };
    int result = 0;
    group.reader.set_fill(true);  // the drive load counts its bytes as fill bytes too
    try {
      // A demand's batch size, and no publish target since a fill has no device readiness words.
      result = group.reader.read(
          fill_row_,
          fill_experts_,
          fill_slots_,
          kBounceRows,
          [](size_t) { return false; },
          nullptr,
          &packed,
          SIZE_MAX,
          advance,
          nullptr);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "ERROR %sprefill fill: %s\n", error_prefix<Layout>().c_str(), error.what());
      result = 0;
    }
    group.reader.set_fill(false);
    _mm_sfence();  // the rows' bytes land before the caller is told (fill_landed_)
    // No tier state here: until the owner's epilogue the claimed slots stay filling, so no admission takes one and
    // release() refuses it.
    fill_result_ = result;
    if (result == 1) {
      fill_landed_.store(static_cast<int64_t>(fill_slots_.size()), std::memory_order_release);
    } else {
      advance();
    }
    end_busy(group);
    fill_state_.store(result == 1 ? kFillOk : kFillFailed, std::memory_order_release);
  }

  // The fill's epilogue, on the owner after the join (fill_join), since the fill thread never writes the tier. Clears
  // every claimed slot's filling flag; a failed fill releases (unmaps) its rows that did not land and counts the read
  // error and the map change on the owner's counters.
  void finish_fill_owned() {
    fill_unfinished_ = false;
    Tier& tier = tiers_[fill_row_];
    for (size_t i = 0; i < fill_slots_.size(); ++i) {
      const int64_t slot = fill_slots_[i];
      tier.filling[slot] = 0;
      if (fill_result_ != 1 && !(i < fill_packed_.size() && fill_packed_[i] != 0)) release_locked(fill_row_, slot);
    }
    if (fill_result_ != 1) {
      count<kReadErrors>();
      count<kVersion>();
    }
  }

  // Updates the host mirror (mapping()) only, for a change the device learns from a record's delta.
  //
  // With the GPU drip, also bumps the entry's generation after the store, and fences before any caller can rewrite a
  // RAM slot it unmapped here: a drip that read the old mapping then sees a new generation once its copy completed
  // (GpuDrip::source_unchanged), had the rewrite begun under it.
  void publish_mirror(int64_t row, int64_t expert, int32_t slot) {
    __atomic_store_n(map_ + row * experts_ + expert, slot, __ATOMIC_RELEASE);
    if (drip_gen_ != nullptr) {
      drip_gen_[row * experts_ + expert].fetch_add(1, std::memory_order_release);
      std::atomic_thread_fence(std::memory_order_seq_cst);
    }
  }

  // Records an eager path's change: the mirror, and the bulk delta the device applies before the next forward.
  void publish_map(int64_t row, int64_t expert, int32_t slot) {
    publish_mirror(row, expert, slot);
    bulk_.push_back({static_cast<int32_t>(row), static_cast<int32_t>(expert), slot});
  }

  // Validates group g's split table (one entry per n = 0..Wire::kLanes, 0 <= split[n] <= n) and publishes it to the
  // device, in its node's table, with relaxed stores; the device reads each entry once per post.
  void store_split(int g, const int64_t* split, int64_t count) {
    if (count != static_cast<int64_t>(Wire::kLanes) + 1)
      throw std::runtime_error(error_prefix<Layout>() + "the CPU split table has one entry per n = 0..Wire::kLanes");
    for (int64_t n = 0; n < count; ++n)
      if (split[n] < 0 || split[n] > n)
        throw std::runtime_error(error_prefix<Layout>() + "the CPU split table must satisfy 0 <= split[n] <= n");
    for (int64_t n = 0; n < count; ++n)
      __atomic_store_n(
          reinterpret_cast<int32_t*>(lease_ + Wire::kSplit + g * Wire::kSplitStride) + n,
          static_cast<int32_t>(split[n]),
          __ATOMIC_RELAXED);
  }

  // Ends the slot's prefill ownership: decode used it, or it is gone. `own` is the row of the group whose range holds
  // the slot.
  void disown_locked(GroupRow& own, Tier& tier, int64_t slot) {
    if (tier.prefill_owned[slot]) {
      tier.prefill_owned[slot] = 0;
      --own.owned;
    }
  }

  // The row of the group whose range holds `slot`. Only a slot some group owns is ever prefill-owned.
  GroupRow& owner_row(int64_t row, int64_t slot) {
    for (int g = 0; g < dist_.size(); ++g)
      if (dist_.group(g).owns(row, slot)) return dist_.group(g).rows[row];
    throw std::logic_error(
        error_prefix<Layout>() + "slot " + std::to_string(slot) + " of row " + std::to_string(row) +
        " belongs to no NUMA group");
  }

  // disown_locked for a slot whose group is not at hand (a seam slot no group owns is never prefill-owned).
  void disown_slot_locked(int64_t row, Tier& tier, int64_t slot) {
    if (tier.prefill_owned[slot]) disown_locked(owner_row(row, slot), tier, slot);
  }

  // Takes a slot for a row a Python-side read admits (assign, and the prefill fill path), in the range of `expert`'s
  // home group.
  //
  // With a prefill share set, no free slot in the range, and the layer already holding that many prefill-owned rows
  // (every group's), the victim is the LRU owned row of the range, under every exclusion of take_slot_locked, so a
  // prefill displaces at most `share` of decode's rows. Otherwise, and whenever no owned row qualifies, it is
  // take_slot_locked's choice (a free slot first); with `stop_at_share` and no qualifying owned row it is none (-1).
  // Under a share the slot becomes prefill-owned.
  int64_t take_admit_slot_locked(
      int64_t row,
      int64_t expert,
      std::span<const int32_t> protect,
      bool fallback,
      int64_t* evicted,
      bool stop_at_share = false) {
    Tier& tier = tiers_[row];
    GroupRow& own = dist_.home_group(expert).rows[row];
    int64_t slot = -1;
    const int64_t share = prefill_share_.load(std::memory_order_relaxed);  // once: one admission, one share
    if (share > 0 && owned_in_row(row) >= share && no_free_slot_locked(tier, own)) {
      int64_t best = -1;
      for (int64_t s = own.lo; s < own.hi; ++s) {
        if (!tier.prefill_owned[s] || tier.state[s] != kReady || tier.filling[s]) continue;
        const int32_t holder = tier.slot_to_expert[s];
        if (tier.hot[holder] || listed(protect, holder)) continue;
        if (best < 0 || tier.stamp[s] < tier.stamp[best]) best = s;
      }
      if (best >= 0) {
        *evicted = tier.slot_to_expert[best];
        release_locked(row, best);  // unmaps it before its bytes are overwritten, and ends its ownership
        count<kEvictions>();
        count<kEagerEvictions>();
        slot = best;
      } else if (stop_at_share) {
        *evicted = -1;
        return -1;
      }
    }
    if (slot < 0) slot = take_slot_locked(row, expert, protect, fallback, evicted);
    if (slot >= 0 && share > 0) {
      tier.prefill_owned[slot] = 1;
      ++own.owned;
    }
    return slot;
  }

  // The prefill-owned rows of the layer, over every group.
  int64_t owned_in_row(int64_t row) const {
    int64_t owned = 0;
    for (int g = 0; g < dist_.size(); ++g)
      owned += dist_.group(g).rows[row].owned;
    return owned;
  }

  // True when no slot of the group's range is kFree (a seam slot no group owns is kFree forever, so the range alone
  // decides).
  static bool no_free_slot_locked(const Tier& tier, const GroupRow& own) {
    for (int64_t slot = own.lo; slot < own.hi; ++slot)
      if (tier.state[slot] == kFree) return false;
    return true;
  }

  // An eager admission's slot, in the range of `expert`'s home group: a kFree one, else the LRU kReady one that is not
  // hot, protected or filling, which is evicted; never kStaging or kSpec. With `fallback`, a protected slot is evicted
  // when nothing else qualifies. Returns -1 when no slot can be taken; `*evicted` is the displaced expert or -1.
  int64_t take_slot_locked(
      int64_t row, int64_t expert, std::span<const int32_t> protect, bool fallback, int64_t* evicted) {
    Tier& tier = tiers_[row];
    const GroupRow& own = dist_.home_group(expert).rows[row];
    *evicted = -1;
    for (int64_t slot = own.lo; slot < own.hi; ++slot) {
      if (tier.state[slot] == kFree) return slot;
    }
    int64_t best = -1;
    int64_t spare = -1;
    for (int64_t slot = own.lo; slot < own.hi; ++slot) {
      if (tier.state[slot] != kReady) continue;
      if (tier.filling[slot]) continue;  // a prefill fill is still writing it
      const int32_t holder = tier.slot_to_expert[slot];
      if (tier.hot[holder]) continue;
      if (listed(protect, holder)) {
        if (spare < 0 || tier.stamp[slot] < tier.stamp[spare]) spare = slot;
        continue;
      }
      if (best < 0 || tier.stamp[slot] < tier.stamp[best]) best = slot;
    }
    if (best < 0 && fallback) best = spare;
    if (best < 0) {
      count<kNoVictim>();
      return -1;
    }
    const int32_t victim = tier.slot_to_expert[best];
    publish_map(
        row, victim, -1);  // unmapped before its bytes are overwritten, so the device never reads a half-rewritten row
    tier.expert_slot[victim] = -1;
    tier.slot_to_expert[best] = -1;
    tier.state[best] = kFree;
    disown_locked(dist_.home_group(expert).rows[row], tier, best);
    *evicted = victim;
    count<kEvictions>();
    count<kEagerEvictions>();
    return best;
  }

  // Frees `slot` and unmaps its expert, if any.
  void release_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const int32_t expert = tier.slot_to_expert[slot];
    if (expert >= 0) {
      publish_map(row, expert, -1);
      tier.expert_slot[expert] = -1;
    }
    tier.slot_to_expert[slot] = -1;
    tier.state[slot] = kFree;
    disown_slot_locked(row, tier, slot);
  }

  // Marks the expert's RAM row as just used, if it is READY. `group` is the expert's home.
  void stamp_routed_locked(Group& group, int64_t row, Tier& tier, int32_t expert) {
    const int32_t slot = tier.expert_slot[expert];
    if (slot >= 0 && tier.state[slot] == kReady) {
      tier.stamp[slot] = ++group.tick;
      disown_locked(group.rows[row], tier, slot);
    }
  }

  // Serves a record with no lanes: refreshes the recency of the group's own routed experts' RAM rows (another group's
  // are its own to stamp: ticks are per group, and so is the ownership count). No eviction, no read.
  void touch_request(Group& group, const Request& request) {
    if (request.row < 0 || request.row >= layers_) return;
    Tier& tier = tiers_[request.row];
    for (int32_t expert : request.protect) {
      if (expert >= 0 && expert < experts_ && Wire::home(expert) == group.index)
        stamp_routed_locked(group, request.row, tier, expert);
    }
  }

  // Picks a decode miss's RAM victim from the group's range: a free slot first, else the least recently used READY slot
  // that is not VRAM-hot, not routed by this record (`wanted`), and not filling. Its expert is unmapped on the host
  // mirror (the device learns it from the delta) and returned in `old` (-1 for a free slot). Returns -1 when none
  // qualifies: the miss is read into its staging slot but not cached.
  // `count_eviction` false: a promotion's pick, run by another group's thread, which must not count into this block.
  int64_t take_victim_locked(
      Group& group, int64_t row, std::span<const int32_t> wanted, int32_t* old, bool count_eviction = true) {
    Tier& tier = tiers_[row];
    GroupRow& own = group.rows[row];
    *old = -1;
    for (int64_t slot = own.lo; slot < own.hi; ++slot) {
      if (tier.state[slot] == kFree) return slot;
    }
    int64_t best = -1;
    for (int64_t slot = own.lo; slot < own.hi; ++slot) {
      if (tier.state[slot] != kReady || tier.filling[slot]) continue;
      const int32_t expert = tier.slot_to_expert[slot];
      if (tier.hot[expert] || listed(wanted, expert)) continue;
      if (best < 0 || tier.stamp[slot] < tier.stamp[best]) best = slot;
    }
    if (best < 0) return -1;
    *old = tier.slot_to_expert[best];
    publish_mirror(row, *old, -1);
    tier.expert_slot[*old] = -1;
    tier.slot_to_expert[best] = -1;
    disown_locked(own, tier, best);
    if (count_eviction) count<kEvictions>(group);
    return best;
  }

  // A demand miss took `victim`: evict carries a = the evicted expert (-1 for a free slot), b = its age in the group's
  // use ticks (each routed lane's use is one tick), c = the expert moving in. Before the miss restamps the slot.
  void trace_evict(Group& group, const Tier& tier, const Request& request, int64_t victim, int32_t old, int32_t incoming)
    requires(Build::kMetrics)
  {
    const int64_t age = old >= 0 ? static_cast<int64_t>(group.tick - tier.stamp[victim]) : -1;
    group.trace.emit("evict", request.row, request.gen, request.seq, group.index, old, age, incoming);
  }

  // What serve_record's phases hand on: the copy job, the misses in lane order, and every expert the record routes.
  // Fixed-capacity members keep the record path free of allocation.
  struct RecordPlan {
    int64_t idx = 0;  // the record's index in the demand ring
    CopyJob job;
    FixedVec<int32_t, kWanted> wanted;
    FixedVec<int32_t, Wire::kLanes> missing;
    FixedVec<int64_t, Wire::kLanes> slots;
    FixedVec<int32_t, Wire::kLanes> miss_lane;  // the lane index of each miss
    Wire::LaneMask pooled = 0;  // bit i: miss i was swapped in from the speculative pool and is not read
    // With a pooled miss, the rows read_misses reads, in miss order, and each miss's read ordinal (-1: pooled).
    FixedVec<int32_t, Wire::kLanes> read_experts;
    FixedVec<int64_t, Wire::kLanes> read_slots;
    FixedVec<int32_t, Wire::kLanes> read_lanes;
    FixedVec<int32_t, Wire::kLanes> ordinal;
    // InstrBuild trace (miss_land): unforced miss i's pool entry when planned, a PoolSeen.
    int8_t pool_seen[Wire::kLanes] = {};
  };
  enum PoolSeen : int8_t { kPoolSeenNone = 0, kPoolSeenLanded = 1, kPoolSeenReading = 2 };

  // The record's CPU-miss jobs still to submit, advanced as rows land.
  struct CpuMissBatch {
    Wire::LaneMask sent = 0;  // bit i: miss i went to the CPU
    int left = 0;
    uint32_t next = 0;  // the next CPU-miss job's sequence, short of the last (job.late_seq)
    // Two-stage: the staged jobs whose second stage is not open yet, each one's sequence and the misses still landing.
    uint32_t staged_seq[Wire::kLanes] = {};
    Wire::LaneMask staged_rows[Wire::kLanes] = {};
    int staged = 0;
  };

  // Fail-stops with the record's identity and `why`.
  [[noreturn]] void fail_record(const Request& request, const std::string& why) {
    fail_stop(
        error_prefix<Layout>() + "request " + std::to_string(request.seq) + " of row " + std::to_string(request.row) +
        ": " + why);
  }

  // Collects the record's routed experts into plan->wanted, each once: the protect list (stamped as routed), then the
  // lanes'. Only the group's own experts are stamped. Fail-stops on an out-of-range expert or one a lane names twice.
  void collect_wanted_locked(Group& group, Tier& tier, const Request& request, RecordPlan* plan) {
    for (int32_t expert : request.protect) {
      if (expert < 0 || expert >= experts_) fail_record(request, "an expert is out of range");
      if (!listed(plan->wanted, expert)) plan->wanted.push_back(expert);
      if (Wire::home(expert) == group.index) stamp_routed_locked(group, request.row, tier, expert);
    }
    FixedVec<int32_t, Wire::kLanes> lane_experts;
    for (const Lane& lane : request.lanes) {
      if (lane.expert < 0 || lane.expert >= experts_) fail_record(request, "an expert is out of range");
      if (listed(lane_experts, lane.expert)) fail_record(request, "a lane names an expert twice");
      lane_experts.push_back(lane.expert);
      if (!listed(plan->wanted, lane.expert)) plan->wanted.push_back(lane.expert);
    }
  }

  // Checks every lane against the tier and fail-stops on any disagreement. A miss goes to the read lists, a copy-engine
  // or CPU hit (stamped) to the copy job, and an SM hit nowhere. Then checks the map chain: the row's next one when the
  // record misses, else 0.
  void classify_lanes_locked(Group& group, Tier& tier, const Request& request, RecordPlan* plan) {
    const bool host_lanes = copy_engine_ != nullptr && copy_armed_.load(std::memory_order_acquire) && request.captured;
    const bool cpu_row = host_lanes && group.cpu != nullptr && group.cpu->eligible(request.row);
    CopyJob& job = plan->job;
    GroupRow& own = group.rows[request.row];
    const bool judging = own.promoted_chain != 0;  // the row's last promotion is judged on this record
    for (size_t j = 0; j < request.lanes.size(); ++j) {
      const Lane& lane = request.lanes[j];
      if (Wire::home(lane.expert) != group.index) continue;  // another group's lane
      // a = the expert, b = its kind | 256 when spilled (no VRAM destination), c = its lane: expert_breakdown.py.
      if constexpr (Build::kMetrics)
        group.trace.emit("lane_kind", request.row, request.gen, request.seq, group.index, lane.expert,
                         static_cast<int64_t>(lane.kind) | (lane.dst < 0 ? 256 : 0), static_cast<int64_t>(j));
      const auto fail = [&](const char* why) {  // the message is built only to fail: the hot path allocates nothing
        fail_record(
            request,
            "lane " + std::to_string(j) + " (expert " + std::to_string(lane.expert) + ", slot " +
                std::to_string(lane.slot) + ")" + why);
      };
      if (lane.kind == Wire::kKindHitScratch) {
        // A landed GPU drip: C1s copies it from its scratch slot, which holds this row's expert until every record
        // that could have read its word is handled (GpuDrip::holds).
        const Drip* drip = drip_.load(std::memory_order_acquire);
        if (!host_lanes || drip == nullptr || !drip->holds(lane.slot, request.row, lane.expert))
          fail(": a scratch lane whose drip slot does not hold the expert");
        count<kDripUsed>(group);
        if constexpr (Build::kMetrics)
          group.trace.emit("drip_use", request.row, request.gen, request.seq, group.index, lane.expert, lane.slot);
        continue;
      }
      if (is_miss(lane.kind)) {
        // Only reserve_staging(0) leaves a reserved row no staging slot: every expert is mapped, so a miss is a bug.
        if (own.staging_reserved && own.staging.empty())
          fail(" is a RAM miss on a row with no staging slots (staging is off: every streamed layer is fully pinned)");
        if (lane.slot < 0) {
          // A forced CPU miss (spill): no staging slot; reserve_victims_locked reads it into a RAM victim or swaps
          // in its pool row.
          if (lane.kind != Wire::kKindMissCpu) fail(" is a GPU miss without a staging slot");
        } else {
          if (!listed(own.staging, lane.slot) || tier.state[lane.slot] != kStaging) fail(" is not a staging slot");
          if (listed(plan->slots, static_cast<int64_t>(lane.slot))) fail(" shares its staging slot with another miss");
        }
        if (tier.expert_slot[lane.expert] >= 0) fail(" misses an expert the tier holds");
        if (lane.kind == Wire::kKindMissCpu) {
          if (!cpu_row || group.cpu->parts() < 2) fail(": a CPU miss on a row without CPU experts' miss part");
          ++job.late_cpu;
          // a = the expert, b = its lane in the record: joins a forward's misses to its routes (verify_split.py).
          if constexpr (Build::kMetrics)
            group.trace.emit("miss_expert", request.row, request.gen, request.seq, group.index, lane.expert,
                             static_cast<int64_t>(j));
        }
        if (judging && listed(own.promote_evicted, lane.expert)) count<kSpecPromoteVictimRouted>(group);
        plan->missing.push_back(lane.expert);
        plan->slots.push_back(lane.slot);
        plan->miss_lane.push_back(static_cast<int32_t>(j));
        continue;
      }
      if (lane.slot < 0 || lane.slot >= tier.capacity || !group.owns(request.row, lane.slot) ||
          tier.expert_slot[lane.expert] != lane.slot || tier.state[lane.slot] != kReady) {
        fail(": the device maps it there, the tier does not");
      }
      tier.stamp[lane.slot] = ++group.tick;
      disown_locked(own, tier, lane.slot);
      if (judging && listed(own.promoted, lane.expert)) count<kSpecPromoteUsed>(group);
      if (lane.kind == Wire::kKindHitCopy) {
        if (!host_lanes || !copy_engine_->eligible(request.row, lane.dst))
          fail(": a copy-engine lane on an ineligible row");
      } else if (lane.kind == Wire::kKindHitCpu) {
        if (!cpu_row) fail(": a CPU lane on a row without CPU experts");
        job.cpu_mask |= Wire::LaneMask{1} << j;
      } else {
        continue;  // Wire::kKindHitSm: the device's SM kernel copies it, the host has nothing to do
      }
      job.lanes[job.count++] = CopyLane{static_cast<int32_t>(j), lane.slot, lane.dst, lane.weight};
      job.mask |= Wire::LaneMask{1} << j;
    }
    if (judging) {
      own.promoted.clear();
      own.promote_evicted.clear();
      own.promoted_chain = 0;
    }
    if (NumaNodeDistributor<Source>::miss_nodes(request) != 0) {
      // A group's own miss is checked against the row's last published delta, which is exact: the device waits for the
      // miss's pieces, so the group reads the record before the delta of its chain can publish, and the device posted
      // it after the delta before it. A group with no miss here may have been lapped (nothing waits on it), so it
      // checks nothing and only follows the chain. At one node every miss record is the group's own and this is the
      // row's chain check, as before.
      if (!plan->missing.empty() && request.chain != dist_.expected_chain(request.row)) {
        fail_record(
            request,
            "map chain " + std::to_string(request.chain) + ", the row expects " +
                std::to_string(dist_.expected_chain(request.row)));
      }
    } else if (request.chain != 0) {
      fail_record(request, "map chain " + std::to_string(request.chain) + " on a record without a miss");
    }
  }

  // Submits to the CPU expert engine; fail-stops if its ring is full.
  void submit_cpu_job(Group& group, const Request& request, const CpuJob& cpu_job) {
    if (!group.cpu->submit(cpu_job)) fail_record(request, "the CPU expert ring is full");
    this->template count<kCpuJobs>(group);
  }

  // Submits the record's CPU-hit job, then its copy job, so the CPU starts first. Returns the CPU misses' batch, whose
  // jobs read_misses submits as their rows land.
  CpuMissBatch submit_host_lanes(Group& group, const Request& request, RecordPlan* plan) {
    const SvcTimer timer(svc_[group.index].cpu_ns);
    CopyJob& job = plan->job;
    CpuMissBatch misses;
    misses.left = job.late_cpu;
    if (job.count == 0 && job.late_cpu == 0) return misses;
    job.gen = request.gen;
    job.idx = plan->idx;
    job.row = request.row;
    job.group = group.index;
    job.groups = NumaNodeDistributor<Source>::host_nodes(request);
    if constexpr (Build::kMetrics) job.submit_ns = now_ns();  // copy_latency_ns, a metric
    this->template count<kCopyJobs>(group);
    this->template count<kCopyLanes>(group, job.count + job.late_cpu);
    const int cpu_hits = std::popcount(job.cpu_mask);
    this->template count<kCpuLanes>(group, cpu_hits + job.late_cpu);
    if (cpu_hits > 0 || job.late_cpu > 0) {
      // One sequence per job the record can need: the hits' and one per CPU miss. The last miss job takes the last,
      // so the copy thread's done(late_seq) covers however the misses were batched.
      const uint32_t first = group.cpu->claim((cpu_hits > 0 ? 1 : 0) + job.late_cpu);
      job.cpu_seq = first;
      misses.next = first + (cpu_hits > 0 ? 1u : 0u);
      job.late_seq = misses.next + static_cast<uint32_t>(job.late_cpu) - 1u;
      if (cpu_hits > 0) {
        CpuJob cpu_job;
        cpu_job.row = request.row;
        cpu_job.part = 0;
        cpu_job.seq = first;
        cpu_job.per_token = true;
        for (int i = 0; i < job.count; ++i) {
          const CopyLane& lane = job.lanes[i];
          if ((job.cpu_mask >> lane.lane & 1u) == 0) continue;
          cpu_job.slots[cpu_job.k] = lane.host_slot;
          cpu_job.weights[cpu_job.k] = lane.weight;
          cpu_job.lanes[cpu_job.k] = lane.lane;
          ++cpu_job.k;
        }
        submit_cpu_job(group, request, cpu_job);
      }
    }
    copy_engine_->submit(group.index, job);
    return misses;
  }

  // An unforced miss reads NVMe even when its group's pool of the row holds the expert (DSV41_REFERENCE.md
  // section 33.36); counts those by the entry's state and the lane's kind, and returns that state.
  PoolSeen count_skipped_pooled(Group& group, int64_t row, const Lane& lane) {
    const int i = pool_->find(row, group.index, lane.expert);
    if (i < 0) return kPoolSeenNone;
    const bool landed =
        pool_state(pool_->entry(row, group.index, i).word.load(std::memory_order_acquire)) == kPoolLanded;
    const bool cpu = lane.kind == Wire::kKindMissCpu;
    if (landed)
      cpu ? count<kSpecSkipLandedCpu>(group) : count<kSpecSkipLandedGpu>(group);
    else
      cpu ? count<kSpecSkipReadingCpu>(group) : count<kSpecSkipReadingGpu>(group);
    return landed ? kPoolSeenLanded : kPoolSeenReading;
  }

  // Unforced GPU miss i with pool_serve_, whose row landed in its group's pool: S streams it from the pool slot, so it
  // is neither read nor copied. With a victim the pool slot becomes the expert's and the victim is pooled until the
  // delta publishes (take_pooled_locked's swap); with none the row is not cached and the entry stays swapped until the
  // group's next record (release_streamed). Publishes the lane's PieceSource word, then every piece bit. Returns the
  // pool slot, or -1 when the row is not landed there: one still reading is read again, never waited for.
  int32_t point_pooled_locked(Group& group, const Request& request, const RecordPlan& plan, size_t i, int64_t victim) {
    const int g = group.index;
    const int32_t expert = plan.missing[i];
    const int e = pool_->find(request.row, g, expert);
    if (e < 0) return -1;
    PoolEntry& entry = pool_->entry(request.row, g, e);
    int32_t slot;
    {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      if (entry.word.load(std::memory_order_acquire) != pool_word(kPoolLanded, expert)) return -1;
      slot = entry.slot;
      if (victim >= 0) {
        entry.slot = static_cast<int32_t>(victim);
        entry.freed_at = request.chain;
      } else {
        entry.freed_at = kPoolStreaming;
        streamed_[g].push_back(StreamedEntry{request.row, e});
      }
      entry.word.store(pool_word(kPoolSwapped, kPoolNoExpert), std::memory_order_release);
    }
    // The row's bytes were ordered by the landed word's acquire; the source word goes before the bits that admit it.
    uint8_t* const line = piece_word_at(plan.idx, plan.miss_lane[i]);
    store_release64(line + Wire::kLeasePieceSource, piece_source_word(request.gen, slot));
    store_release64(line, piece_word(request.gen) | ((uint64_t{1} << kPieces) - 1));
    _mm_sfence();
    count<kSpecPoolServed>(group);
    // a = expert, b = the pool slot S streams from, c = the victim it pooled, -1 when the row is not cached.
    if constexpr (Build::kMetrics)
      group.trace.emit("pool_point", request.row, request.gen, request.seq, g, expert, slot, victim);
    return slot;
  }

  // At group g's next posted record: S of every earlier record ended (the device posts one only after the previous
  // chain ended), so the entries it streamed uncached rows from are empty again.
  void release_streamed(int g) {
    std::lock_guard<std::mutex> lock(pool_->mutex(g));
    for (const StreamedEntry& s : streamed_[g])
      pool_->entry(s.row, g, s.entry).word.store(kPoolEmpty, std::memory_order_release);
    streamed_[g].clear();
  }

  // The readiness word of record lane `lane` (init_piece_words_locked's).
  uint8_t* piece_word_at(int64_t idx, int32_t lane) {
    return lease_ + Wire::kLeasePieceMask +
           (idx * Wire::kLanes + static_cast<int64_t>(lane)) * Wire::kLeasePieceMaskLineBytes;
  }

  // A forced miss's pool row: when group g's pool of the row holds `expert` landed, returns the entry's slot and gives
  // the entry `victim`, swapped until the record's delta publishes (release_swapped); else -1 and the miss is read.
  // An entry still reading the expert is waited for first. Takes the group's pool mutex itself.
  int32_t take_pooled_locked(Group& group, const Request& request, int32_t expert, int32_t victim) {
    const int g = group.index;
    const int i = pool_->find(request.row, g, expert);
    if (i < 0) return -1;
    PoolEntry& entry = pool_->entry(request.row, g, i);
    if (entry.word.load(std::memory_order_acquire) == pool_word(kPoolReading, expert)) {
      // Promotion: the demand waits for the one speculative read in flight rather than read the row again; the
      // service's busy episode times the wait. The caller holds the row's tier state meanwhile, for at most that one
      // row read.
      count<kSpecPromoted>(group);
      entry.boost.store(1, std::memory_order_release);  // an idle-drive read drops its gate for this row
      while (entry.word.load(std::memory_order_acquire) == pool_word(kPoolReading, expert))
        _mm_pause();
    }
    int32_t slot;
    [[maybe_unused]] uint32_t for_seq;  // read by the InstrBuild event only
    {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      if (entry.word.load(std::memory_order_acquire) != pool_word(kPoolLanded, expert)) return -1;
      slot = entry.slot;
      for_seq = entry.for_seq.load(std::memory_order_relaxed);
      entry.slot = victim;
      entry.freed_at = request.chain;
      entry.word.store(pool_word(kPoolSwapped, kPoolNoExpert), std::memory_order_release);
      count<kSpecUsed>(group);
    }
    if constexpr (Build::kMetrics) {
      // seq = the demand record that swapped it in, a = expert, b = slot, c = the entry's source seq (spec_submit's seq).
      if (spec_ != nullptr)  // the pool exists without the speculative state in some tests
        spec_->groups[g]->trace.emit("spec_use", request.row, request.gen, request.seq, g, expert, slot, for_seq);
    }
    return slot;
  }

  template <Counter K>
  void spec_count(SpecGroup& spec, int64_t n = 1) {
    count_into<K>(spec.core, n);
  }

  // Service thread, before any whole record is handled: notes its row's seq, so a job whose source or target row is in
  // service is stale. The fence pairs with spec_read's: a forced miss either sees the pool read and promotes it, or
  // spec_read sees this seq and drops the read.
  void note_serving(Group& group, const Request& request) {
    if (request.row < 0 || request.row >= layers_) return;
    spec_->groups[group.index]->row_seq[request.row].store(request.seq, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);
  }

  // Service thread, after any whole record: hands it to the group's speculative thread when its row has a target and,
  // with the CPU scorer, it staged a CPU input (the GPU scorer scores every record). A full ring drops it: the service
  // never waits.
  void offer_spec(Group& group, const Request& request) {
    if (request.row < 0 || request.row >= layers_) return;
    SpecGroup& spec = *spec_->groups[group.index];
    if (spec_->config.target[request.row] < 0) return;
    int64_t tokens = 1;
    if (spec_->config.candidates == nullptr) {
      bool staged = false;
      for (const Lane& lane : request.lanes)
        staged |= lane.kind == Wire::kKindHitCpu || lane.kind == Wire::kKindMissCpu;
      if (!staged) return;
      if (spec_->tokens_max > 1) {
        // The live tokens, not the record's: a verify's dead rows are copies of the anchor's input.
        const uint8_t* table = spec_->x_base + request.row * spec_->x_stride + spec_->tokens_max * spec_->x_token_bytes;
        uint32_t live;
        std::memcpy(&live, table + CpuTokenTable::kLiveTokens, 4);
        tokens = live;
        if (tokens < 1 || tokens > spec_->tokens_max) {
          count<kSpecDropped>(group);
          return;
        }
      }
    }
    if (!spec.ring.push(SpecJob{request.seq, request.row, tokens})) {
      count<kSpecDropped>(group);
      return;
    }
    spec.bell.ring();
  }

  // True once the group served a record of `target`, or of the source row (a lapped target record leaves no trace),
  // after `source_seq`: the read would land behind the record it was for. Each group judges by its own row_seq, so a
  // job in flight can be stale for one group and live for the other: the layer then reads fewer than per_layer rows,
  // never more.
  static bool spec_stale(const SpecGroup& spec, int64_t source, int64_t target, uint32_t source_seq) {
    const uint32_t next = skip_zero(source_seq + 1u);
    for (const int64_t row : {source, target}) {
      const uint32_t last = spec.row_seq[row].load(std::memory_order_acquire);
      if (last != 0 && reached(last, next)) return true;
    }
    return false;
  }

  // The scorer's skips for `target`: an expert VRAM-hot there, mapped there (the host mirror), or pooled from another
  // source record. Lock-free loads of words other groups' threads write; a race costs at most a boundary candidate.
  void fill_spec_skip(int64_t target, uint32_t source_seq, uint8_t* skip) {
    Tier& tier = tiers_[target];
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const bool hot = std::atomic_ref<uint8_t>(tier.hot[expert]).load(std::memory_order_relaxed) != 0;
      const bool mapped = __atomic_load_n(map_ + target * experts_ + expert, __ATOMIC_ACQUIRE) >= 0;
      skip[expert] = hot || mapped || pool_->pooled_before(target, static_cast<int32_t>(expert), source_seq);
    }
  }

  // Speculative thread (`threaded`) or spec_pump: scores `job`'s target row and reads this group's picks into the
  // pool. Both groups compute the same list from the same input and state, so the layer's budget holds over both.
  void serve_spec_job(int g, const SpecJob& job, bool threaded) {
    SpecGroup& spec = *spec_->groups[g];
    const RamPrefetchConfig& config = spec_->config;
    const int64_t target = config.target[job.row];
    if (spec_stale(spec, job.row, target, job.seq)) {
      spec_count<kSpecDropped>(spec);
      return;
    }
    if (config.candidates != nullptr) {
      serve_gpu_job(g, job, target, threaded);
      return;
    }
    int64_t start = 0;
    if constexpr (Build::kMetrics) start = now_ns();
    fill_spec_skip(target, job.seq, spec.skip.data());
    const int64_t gate = config.gate[job.row];
    int32_t chosen[GateScorer::kMaxPerLayer];
    int32_t ranks[GateScorer::kMaxPerLayer];
    float margins[GateScorer::kMaxPerLayer];
    const int n = spec.scorer.choose(
        spec_->x_base + job.row * spec_->x_stride, job.tokens, spec_->x_token_bytes,
        config.gates + gate * experts_ * config.hidden, config.bias + gate * experts_, experts_, config.hidden,
        config.top_k, config.per_token, config.per_layer, spec.skip.data(), chosen, config.top_k_only,
        Build::kMetrics ? ranks : nullptr, Build::kMetrics ? margins : nullptr);
    if constexpr (Build::kMetrics) {
      stats_.add(kSpecScored);
      stats_.add(kSpecScoreNs, now_ns() - start);
    }
    for (int i = 0; i < n; ++i) {
      if (Wire::home(chosen[i]) != g) continue;
      if (threaded && !spec_wait_unheld(spec)) return;  // a quiescer waits for one row, not the whole job
      uint64_t pick = 0;  // spec_submit's gen: rank << 32 | the margin's fp32 bits
      if constexpr (Build::kMetrics) pick = (static_cast<uint64_t>(ranks[i]) << 32) | std::bit_cast<uint32_t>(margins[i]);
      spec_read(g, job.row, target, job.seq, chosen[i], pick);
    }
  }
