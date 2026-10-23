// RamTier members: prefill fills, the copy engine and its wait gate, the GPU drip, and the CPU experts.
// A class-body fragment: ram_tier.h includes it inside RamTier, at the place these members always sat. Include
// ram_tier.h, never this file.
#pragma once

  // ---- Prefill fills (SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS) ----
  //
  // A prefill fill reads the rows of many experts into RAM slots on a helper thread while the caller gathers.
  //   - fill_begin: a caller that owns the tier claims slots for `experts` of `row`, in order, until one cannot be
  //   taken,
  //     and starts the helper, which reads the claimed rows through the service's reader.
  //   - A claimed slot is kReady and mapped at once, as assign() leaves it, and flagged `filling` until the read ends:
  //     no admission can evict it and release() refuses it.
  //   - The read's progress publishes how many rows have landed, as a prefix of the claim order (fill_wait).
  //   - The helper holds a busy episode, so the watchdog aborts a hung fill the way it aborts a hung demand.
  //   - fill_end() joins the helper, and so does resume(), so the service thread and a fill never use the reader at
  //   once.
  //   - The helper touches no tier state: the claimed slots stay `filling`, and a failed fill's unlanded rows stay
  //     mapped, until the owner joins it and runs the epilogue (fill_join, finish_fill_owned).

  // Returns the count claimed; slots[i] is expert i's slot. Sets `*evictions` to the number of rows displaced. A
  // prefetch (no `fallback`) stops at the prefill share.
  int64_t fill_begin(
      int64_t row,
      const std::vector<int32_t>& experts,
      const std::vector<int32_t>& protect,
      bool fallback,
      int64_t* slots,
      int64_t* evictions) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("a prefill fill");
    if (fill_thread_.joinable()) throw std::runtime_error(error_prefix<Layout>() + "a prefill fill is already running");
    std::vector<int32_t> claimed;
    std::vector<int64_t> taken;
    *evictions = 0;
    {
      Tier& tier = tiers_[row];
      for (const int32_t expert : experts) {
        if (tier.expert_slot[expert] >= 0) {
          throw std::runtime_error(
              error_prefix<Layout>() + "fill of expert " + std::to_string(expert) + " that holds a slot");
        }
        int64_t evicted = -1;
        // The rest is left to gather_rows' chunked admission, which evicts the share's rows the earlier chunks
        // gathered.
        const int64_t slot =
            take_admit_slot_locked(row, expert, protect, fallback, &evicted, /*stop_at_share=*/!fallback);
        if (slot < 0) break;
        *evictions += evicted >= 0 ? 1 : 0;
        tier.slot_to_expert[slot] = expert;
        tier.state[slot] = kReady;
        tier.stamp[slot] = ++dist_.home_group(expert).tick;
        tier.expert_slot[expert] = static_cast<int32_t>(slot);
        tier.filling[slot] = 1;
        publish_map(row, expert, static_cast<int32_t>(slot));
        claimed.push_back(expert);
        taken.push_back(slot);
      }
      if (!taken.empty()) count<kVersion>();
    }
    for (size_t i = 0; i < taken.size(); ++i)
      slots[i] = taken[i];
    fill_landed_.store(0, std::memory_order_release);
    fill_state_.store(taken.empty() ? kFillOk : kFillRunning, std::memory_order_release);
    if (taken.empty()) return 0;
    fill_row_ = row;
    fill_experts_ = std::move(claimed);
    fill_slots_ = std::move(taken);
    fill_result_ = 0;
    fill_unfinished_ = true;  // the epilogue is owed: fill_join runs it once, on the owner
    fill_thread_ = std::thread([this] { run_fill(); });
    return static_cast<int64_t>(fill_slots_.size());
  }

  // Waits for the fill. Returns 1 once the first `rows` claimed rows have landed, 0 if the fill ended short of them, -1
  // at the deadline.
  int64_t fill_wait(int64_t rows, int64_t timeout_ns) {
    const int64_t deadline = now_ns() + timeout_ns;
    while (true) {
      const int state = fill_state_.load(std::memory_order_acquire);
      if (fill_landed_.load(std::memory_order_acquire) >= rows) return 1;
      if (state != kFillRunning) return 0;  // ended short of `rows`: failed, or never claimed them
      if (now_ns() > deadline) return -1;
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }

  // How many claimed rows have landed so far, a prefix of the claim order (what fill_wait waits on). Never blocks.
  int64_t fill_landed() const {
    return fill_landed_.load(std::memory_order_acquire);
  }

  // Joins the fill and runs its epilogue. Returns 1 when every claimed row landed (or nothing was claimed), 0 when it
  // failed; a failed fill has released its rows that did not land (finish_fill_owned, here on the caller).
  //
  // Runs under caller_mutex_, like every other join of fill_thread_ (resume, stop_thread's final_settle), so no two
  // threads join it at once. An owed epilogue writes the tier, so a caller that does not own it is refused rather than
  // let race the service. A fill starts only on the owner and resume() joins it before handing the tier back, and
  // RamThread::start refuses a tier with a fill owed, so this check is the backstop, not the gate.
  int64_t fill_end() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (fill_unfinished_) require_owner("fill_end with a prefill fill's epilogue owed");
    fill_join();
    return fill_state_.load(std::memory_order_acquire) == kFillFailed ? 0 : 1;
  }

  // Joins the fill thread and runs the owed epilogue. Callers: fill_end, resume() and stop_thread's final_settle (each
  // under caller_mutex_), or the destructor. The join is the synchronization point: every store the fill thread made
  // (fill_result_, the fill's packed flags, the reader's state, the slab bytes) happens-before the epilogue.
  void fill_join() {
    if (fill_thread_.joinable()) fill_thread_.join();
    if (fill_unfinished_) finish_fill_owned();
  }

  // True from fill_begin until the owner's fill_join ran the epilogue. The flag is the owner's, read under
  // caller_mutex_ by a caller that owns the tier (RamThread::start, before the service exists).
  bool fill_owed() const {
    return fill_unfinished_;
  }

  // Sets the number of rows a prefill may own per layer: the share for a prefill, 0 for decode, set before each
  // forward. 0 (the default, and always with SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE off) admits as take_slot_locked always
  // has. Any thread: a relaxed store, read relaxed only by the owner's admissions (a paused caller's, in practice).
  void set_prefill_share(int64_t share) {
    if (share < 0) throw std::runtime_error(error_prefix<Layout>() + "a prefill share cannot be negative");
    prefill_share_.store(share, std::memory_order_relaxed);
  }

  // Copies the hot bitmap of `expected`'s record into the group's hot_scratch (hence not const) and points
  // request->hot_bitmap at it until the next call. Returns false for a missing, overwritten, or malformed record.
  bool read_gpu_hot(Group& group, uint32_t expected, Request* request) {
    if (hot_page_ == nullptr) return false;
    const uint8_t* record = hot_page_ + static_cast<int64_t>((expected - 1u) % Wire::kHotRecords) * hot_stride_;
    if (load_acquire(record) != expected) return false;
    std::vector<uint8_t>& scratch = group.hot_scratch;
    const size_t bytes = scratch.size();
    std::memcpy(scratch.data(), record + Wire::kHotHeaderBytes, bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    asm volatile("" ::: "memory");  // the copy's plain loads must stay before the seq re-check
    if (load_acquire(record) != expected) return false;
    if (experts_ % 8 != 0 && (scratch[bytes - 1] & static_cast<uint8_t>(~((1u << (experts_ % 8)) - 1u))) != 0)
      return false;
    request->hot_bitmap = scratch.data();
    return true;
  }

  // Starts the loads of every line the request's read will touch: the record's lines and its hot record's. The
  // device has just written each, so each is an L3 miss; issued together they overlap instead of queueing behind
  // read_record's branches. Called once the head shows the post, since a line prefetched earlier would be refetched.
  void prefetch_request(const uint8_t* record, uint32_t seq) const {
    for (int64_t line = 0; line < Wire::kRecordBytes; line += 64)
      _mm_prefetch(reinterpret_cast<const char*>(record + line), _MM_HINT_T0);
    if (hot_page_ == nullptr) return;
    const uint8_t* hot = hot_page_ + static_cast<int64_t>((seq - 1u) % Wire::kHotRecords) * hot_stride_;
    for (int64_t line = 0; line < hot_stride_; line += 64)
      _mm_prefetch(reinterpret_cast<const char*>(hot + line), _MM_HINT_T0);
  }

  // Replaces the row's hot set with the record's bitmap, for the group's own experts. Relaxed atomic stores: the RAM
  // prefetch's scorers read every expert's flag from any group's thread (fill_spec_skip).
  void apply_gpu_hot(Group& group, const Request& request) {
    Tier& tier = tiers_[request.row];
    for (int64_t expert = group.index; expert < experts_; expert += Wire::kNodes)
      std::atomic_ref<uint8_t>(tier.hot[expert])
          .store((request.hot_bitmap[expert / 8] >> (expert % 8)) & 1, std::memory_order_relaxed);
  }

  // ---- The copy engine ----

  // Creates the copy engine (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine"), a thread that copies a record's
  // HIT_COPY lanes with the DMA engine. `device` < 0 selects the CPU test backend (HostCopyBackend). Must run before
  // the service thread starts, and the engine stays unarmed until arm_copy_engine(). `wait_timeout_ns` is how long a
  // closed gate may hold the decode stream before the watchdog aborts the process (SGLANG_EXPERT_STREAM_RAM_MISS_TIMEOUT_MS).
  void enable_copy_engine(int64_t device, int64_t spin_ns, int64_t wait_timeout_ns, std::vector<int> cpus) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "enable the copy engine before the service thread starts");
    if (copy_engine_ != nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "the copy engine is already enabled");
    const std::string copy_prefix = std::string(Layout::kName) + " RAM miss copy engine: ";
    std::unique_ptr<CopyBackend> backend;
    if (device < 0) {
      backend = std::make_unique<HostCopyBackend>();
    } else {
      backend = std::make_unique<CudaCopyBackend>(static_cast<int>(device), copy_prefix);
    }
    auto engine = std::make_unique<Engine>(
        std::move(backend),
        layers_,
        dist_.size(),
        spin_ns,
        this,
        copy_prefix,
        std::string(Layout::kName) + "-copy-eng",
        std::move(cpus));
    if (wait_timeout_ns <= 0)
      throw std::runtime_error(error_prefix<Layout>() + "the copy-wait timeout must be positive");
    copy_wait_timeout_ns_ = wait_timeout_ns;
    engine->start();
    copy_engine_ = std::move(engine);
  }

  // ---- The copy wait's gate (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine") ----
  //
  // The decode stream waits (cuStreamWaitValue32) on the lease block's copy gate. The copy wait kernel closes it for
  // record G, with G's seq in the word, when G has copy-engine or CPU lanes. The kernel opens it itself when CopyDone
  // already carries G after the close; otherwise the copy thread does, after it stores CopyDone (copy_completed). Both
  // open with the same word, and the host only by a CAS from G's own closed word, so a stale open for G meets
  // closed(G + k) and changes nothing.

  // The gate word for record `seq`: its sequence number in the high bits, the closed/open state in the low bits.
  static uint32_t gate_word(uint32_t seq, uint32_t low) {
    return channel::gate_word(seq, low);
  }

  // The watchdog's view: the gate word, whose closed bit is set while a copy wait holds the decode stream.
  uint32_t copy_gate() const {
    return channel::gate<TargetChannel>(lease_);
  }

  int64_t copy_wait_timeout_ns() const {
    return copy_wait_timeout_ns_;
  }

  // Teardown (the FFI's stop_thread and close): with no copy thread to follow, a closed copy wait would hold its stream
  // forever. Opening the gate lets the device continue; the consumer traps unless CopyDone is there, which is
  // acceptable because the process is ending either way.
  void open_closed_gate() {
    channel::open_closed_gate<TargetChannel>(lease_);
  }

  // Sets row `row`'s copy table: `count` entries of {source slab address, destination tensor address, row bytes}. Bit i
  // of `sm_mask` leaves entry i to the copy wait's SM reads (SGLANG_EXPERT_STREAM_ENABLE_SM_SMALL_COPIES).
  // Precondition (row_layout.h, ExpertRowLayout): `entries` holds one entry per Layout::kNames name, in kNames order,
  // so sm_mask (a copy-table entry index) and Layout::kSmallMask (a layout-name index bitmask) share one index space.
  void set_copy_table(int64_t row, const int64_t* entries, int64_t count, int64_t dst_rows, int64_t sm_mask) {
    if ((static_cast<uint64_t>(sm_mask) & ~static_cast<uint64_t>(Layout::kSmallMask)) != 0) {
      throw std::runtime_error(
          error_prefix<Layout>() + "sm_mask names a tensor that is not one of the layout's small ones");
    }
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    if (sm_mask < 0 || (count < 63 && (sm_mask >> count) != 0)) {
      throw std::runtime_error(error_prefix<Layout>() + "the SM mask names a copy-table entry that does not exist");
    }
    std::vector<CopyEntry> table;
    for (int64_t i = 0; i < count; ++i) {
      if (entries[3 * i + 2] <= 0) throw std::runtime_error(error_prefix<Layout>() + "a copy-table entry of no bytes");
      table.push_back(
          CopyEntry{
              static_cast<uint64_t>(entries[3 * i]),
              static_cast<uint64_t>(entries[3 * i + 1]),
              entries[3 * i + 2],
              (sm_mask >> i & 1) != 0});
    }
    // The drip copies every entry, SM ones included: C1s has no small-copy split.
    if (drip_owner_ != nullptr) drip_owner_->set_row(row, table);
    copy_engine_->set_table(row, std::move(table), dst_rows);
  }

  // Arms or disarms the copy engine. The device reads Wire::kCopyArmed at every post and types copy-engine and CPU
  // lanes only while it is 1.
  void arm_copy_engine(bool on) {
    if (on && copy_engine_ == nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    copy_armed_.store(on, std::memory_order_release);
    store_release(lease_ + Wire::kCopyArmed, on ? 1u : 0u);
  }

  bool copy_armed() const {
    return copy_armed_.load(std::memory_order_acquire);
  }

  // ---- The GPU drip (gpu_drip.h; analysis/dsv41-drive/LEASE_PROTOCOL.md, "The GPU drip") ----

  // Starts the drip on the copy thread. Needs the copy engine, and runs before the service thread starts: from here on
  // every mirror change bumps the generation the drip checks a source row by (publish_mirror). The scratch
  // (set_drip_scratch) must come before any copy table, which names a row's sources.
  void enable_gpu_drip(DripConfig config) {
    const std::string prefix = error_prefix<Layout>() + "GPU drip: ";
    if (threaded_.load()) throw std::runtime_error(prefix + "enable it before the service thread starts");
    if (copy_engine_ == nullptr) throw std::runtime_error(prefix + "it copies on the copy engine, which is not enabled");
    if (drip_owner_ != nullptr) throw std::runtime_error(prefix + "already enabled");
    if (config.slots < 1 || config.slots > Wire::kScratchSlots)
      throw std::runtime_error(prefix + "1.." + std::to_string(Wire::kScratchSlots) + " scratch slots");
    if (config.picks < 1 || config.picks > SpecCandidates::kMaxDrips)
      throw std::runtime_error(prefix + "1.." + std::to_string(SpecCandidates::kMaxDrips) + " picks per record");
    if (config.chunk_bytes < 1) throw std::runtime_error(prefix + "the chunk must be positive");
    if (!config.target.empty() && static_cast<int64_t>(config.target.size()) != layers_)
      throw std::runtime_error(prefix + "one target per row");
    for (const int32_t t : config.target)
      if (t < -1 || t >= layers_) throw std::runtime_error(prefix + "a target row out of range");
    if (config.candidates != nullptr && config.target.empty())
      throw std::runtime_error(prefix + "the candidate page needs the targets");
    drip_gen_ = std::make_unique<std::atomic<uint32_t>[]>(static_cast<size_t>(layers_ * experts_));
    drip_owner_ = std::make_unique<Drip>(
        std::move(config), this, page_, lease_, map_, drip_gen_.get(), layers_, experts_,
        std::string(Layout::kName) + "-drip");
    drip_.store(drip_owner_.get(), std::memory_order_release);
  }

  // The drip's scratch tensors, one per copy-table entry in its order: {address, bytes of one row}, and its scratch
  // table's address (device memory, Wire::kScratchSlots zeroed u64 words). Once.
  void set_drip_scratch(std::vector<ScratchEntry> entries, uint64_t table) {
    if (drip_owner_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the GPU drip is not enabled");
    drip_owner_->set_scratch(std::move(entries), table);
  }

  // Test hooks (the exports refuse them in ProdBuild): picks offered as a scored record's; holding the chunks back;
  // the slots' {state, row, expert, RAM slot}.
  void drip_offer(int64_t row, std::span<const int32_t> experts) {
    if (drip_owner_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the GPU drip is not enabled");
    drip_owner_->offer(row, experts);
  }
  void drip_hold(bool on) {
    if (drip_owner_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the GPU drip is not enabled");
    drip_owner_->hold(on);
  }
  void drip_view(int64_t* out) const {
    if (drip_owner_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the GPU drip is not enabled");
    drip_owner_->view(out);
  }

  // ---- CPU experts ----

  // Creates group g's CPU expert engine, on that group's cores. The device types a captured post's CPU lanes as the
  // last split[n] of its n eligible lanes (the node's table at Wire::kSplit, written here and by set_cpu_split). The
  // service hands them to the group's CPU expert thread and the copy thread waits for them, so CopyDone covers them.
  // Needs the copy engine; call before the service thread starts.
  void enable_cpu_experts(int g, CpuExpertConfig config, std::vector<int64_t> split) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "enable CPU experts before the service thread starts");
    if (g < 0 || g >= groups())
      throw std::runtime_error(
          error_prefix<Layout>() + "CPU experts name group " + std::to_string(g) + " of " + std::to_string(groups()));
    if (copy_engine_ == nullptr)
      throw std::runtime_error(
          error_prefix<Layout>() + "CPU experts need the copy engine, which completes their lanes");
    std::unique_ptr<CpuExpertEngine>& cpu = dist_.group(g).cpu;
    if (cpu != nullptr)
      throw std::runtime_error(
          error_prefix<Layout>() + "CPU experts are already enabled for group " + std::to_string(g));
    if (config.kernel == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts need the format's kernel");
    if (config.two_stage && dist_.group(g).reader.second_stage() == 0)
      throw std::runtime_error(
          error_prefix<Layout>() + "two-stage CPU misses need row images whose second stage follows the first");
    if (config.two_stage) {
      Source& reader = dist_.group(g).reader;
      const int64_t parts = reader.tables().parts;
      reader.set_span_weights(
          span_weight_units(config.span_weights[0], parts, "SGLANG_EXPERT_STREAM_CPU_TWO_STAGE w13_weights"),
          span_weight_units(config.span_weights[1], parts, "SGLANG_EXPERT_STREAM_CPU_TWO_STAGE w2_weights"));
    }
    if (cpu_kernel_ != nullptr && cpu_kernel_ != config.kernel)
      throw std::runtime_error(
          error_prefix<Layout>() + "every group's CPU experts run one kernel: group " + std::to_string(g) +
          " names " + config.kernel->name() + ", another group " + cpu_kernel_->name());
    cpu_kernel_ = config.kernel;
    config.layers = cpu_layers_.get();
    config.check_calls = Build::kFaults;
    store_split(g, split.data(), static_cast<int64_t>(split.size()));
    const std::string prefix = std::string(Layout::kName) + " CPU experts: ";
    const std::string suffix = Wire::kNodes > 1 ? std::to_string(g) : std::string();
    auto engine =
        std::make_unique<CpuExpertEngine>(std::move(config), prefix, std::string(Layout::kName) + "-cpu-exp" + suffix);
    engine->start();
    cpu = std::move(engine);
    copy_engine_->set_cpu(g, cpu.get());
  }

  // Installs `row`'s layer, made by the enabled kernel, for every group's engine: the layer addresses the whole slab,
  // so one serves them all. Any time, once per row; until then no post types a CPU lane for the row. The layer must
  // hold every slot of the row (make_cpu_layer checks it): forwards take the tier's slots unchecked.
  void set_cpu_layer(int64_t row, const cpu_experts::ExpertLayer& layer) {
    if (cpu_kernel_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    check_cpu_layer_row(row);
    if (layer.kernel != cpu_kernel_)
      throw std::runtime_error(error_prefix<Layout>() + "a CPU expert layer must come from the enabled kernel");
    if (!cpu_layers_->set(row, layer))
      throw std::runtime_error(error_prefix<Layout>() + "a CPU expert layer is registered once");
  }

  // set_cpu_layer of the enabled kernel's make_layer over `shape` and `params` (its std::invalid_argument propagates).
  void make_cpu_layer(int64_t row, const cpu_experts::ExpertLayer& shape, std::span<const std::byte> params) {
    if (cpu_kernel_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    check_cpu_layer_row(row);
    if (shape.capacity < row_capacity(row))
      throw std::runtime_error(
          error_prefix<Layout>() + "CPU expert layer for row " + std::to_string(row) + " holds " +
          std::to_string(shape.capacity) + " slots, the row " + std::to_string(row_capacity(row)));
    set_cpu_layer(row, cpu_kernel_->make_layer(shape, params));
  }

  const cpu_experts::CpuExpertKernel* cpu_kernel() const {
    return cpu_kernel_;
  }

  // The DSpark draft channel on group g's CPU expert engine (CpuExpertEngine::attach_draft): draft_open makes the
  // source, draft_set_layer fills a stage's layer, draft_start attaches it, draft_stop detaches it.
  void draft_open(int g, std::unique_ptr<DraftSource> source, int64_t fatal_wait_ns) {
    draft_cpu(g);
    pending_draft_[g] = std::move(source);
    draft_fatal_ns_[g] = fatal_wait_ns;
  }
  void draft_set_layer(int g, int stage, const cpu_experts::ExpertLayer& layer) {
    DraftSource* d = pending_draft_[g].get();
    if (d == nullptr) throw std::runtime_error(std::string(CpuExpertEngine::kDraftPrefix) + "draft_open first");
    if (stage < 0 || stage >= static_cast<int>(d->layers.size()))
      throw std::runtime_error(std::string(CpuExpertEngine::kDraftPrefix) + "stage " + std::to_string(stage) + " is out of range");
    d->layers[stage] = layer;
  }
  void draft_start(int g) {
    if (pending_draft_[g] == nullptr) throw std::runtime_error(std::string(CpuExpertEngine::kDraftPrefix) + "draft_open first");
    for (size_t s = 0; s < pending_draft_[g]->layers.size(); ++s)
      if (pending_draft_[g]->layers[s].kernel == nullptr)
        throw std::runtime_error(std::string(CpuExpertEngine::kDraftPrefix) + "stage " + std::to_string(s) + " has no layer");
    draft_cpu(g).attach_draft(std::move(pending_draft_[g]), draft_fatal_ns_[g]);
  }
  void draft_stop(int g) {
    draft_cpu(g).detach_draft();
  }
  typename CpuExpertEngine::DraftStats draft_stats(int g) {
    return draft_cpu(g).draft_stats();
  }

  void check_cpu_layer_row(int64_t row) const {
    if (row < 0 || row >= layers_)
      throw std::runtime_error(error_prefix<Layout>() + "CPU expert layer for row " + std::to_string(row) + " of " +
                               std::to_string(layers_));
  }

  // Replaces group g's split table, for re-tuning. Any time; the device reads each entry once per post.
  void set_cpu_split(int g, const int64_t* split, int64_t count) {
    if (cpu_engine(g) == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    store_split(g, split, count);
  }

  // Group g's CPU expert thread's metrics, {jobs, lanes, forward ns, staged jobs, stage-two waits, stage-two wait ns};
  // zeros when its CPU experts are off.
  void cpu_stats(int g, int64_t* out) const {
    const CpuExpertEngine* cpu = cpu_engine(g);
    out[0] = cpu != nullptr ? cpu->jobs() : 0;
    out[1] = cpu != nullptr ? cpu->lanes() : 0;
    out[2] = cpu != nullptr ? cpu->compute_ns() : 0;
    out[3] = cpu != nullptr ? cpu->staged_jobs() : 0;
    out[4] = cpu != nullptr ? cpu->stage_two_waits() : 0;
    out[5] = cpu != nullptr ? cpu->stage_two_wait_ns() : 0;
  }

  // Group g's CPU fit moments (CpuExpertEngine::fit_moments), 2 * kFitMoments int64; zeros when its CPU experts are off.
  void cpu_fit_moments(int g, int64_t* out) const {
    const CpuExpertEngine* cpu = cpu_engine(g);
    if (cpu == nullptr) {
      std::fill(out, out + 2 * CpuExpertEngine::kFitMoments, int64_t{0});
      return;
    }
    cpu->fit_moments(out);
  }

  // Every group's CPU experts' cores, empty without CPU experts. For the caller, before the service thread starts.
  std::vector<int> cpu_cores() const {
    std::vector<int> cores;
    for (int g = 0; g < groups(); ++g) {
      if (dist_.group(g).cpu == nullptr) continue;
      const std::vector<int> own = dist_.group(g).cpu->cores();
      cores.insert(cores.end(), own.begin(), own.end());
    }
    return cores;
  }

  // The copy thread's reason for withholding CopyDone, for the watchdog's abort: "; group G: its CPU job S is not
  // done (...)", with the engine's stall_state, or "; group G: sent no part". Empty at one group, whose abort line stays
  // as it was, or when nothing waits.
  std::string copy_stall() const {
    if (Wire::kNodes == 1 || copy_engine_ == nullptr) return "";
    const uint64_t word = copy_engine_->stall();
    const std::string group = "; group " + std::to_string(word >> 8 & 0xFF);
    switch (word & 0xFF) {
      case Engine::kStallCpu: {
        const CpuExpertEngine* cpu = dist_.group(static_cast<int>(word >> 8 & 0xFF)).cpu.get();
        const uint32_t seq = static_cast<uint32_t>(word >> 32);
        return group + ": its CPU job " + std::to_string(seq) + " is not done" + (cpu ? cpu->stall_state(seq) : "");
      }
      case Engine::kStallNoPart:
        return group + ": sent no part";
      default:
        return "";
    }
  }

  const CpuExpertEngine* cpu_engine(int g) const {
    if (g < 0 || g >= groups()) throw std::runtime_error(error_prefix<Layout>() + "no NUMA group " + std::to_string(g));
    return dist_.group(g).cpu.get();
  }

  // The bytes the DMA moves per expert of `row`: its copy table less the SM entries.
  int64_t copy_expert_bytes(int64_t row) const {
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    return calibration_expert_bytes(copy_engine_->dma_entries(row));
  }

  // Runs the startup calibration of the CPU split (split_calibration.h) and fills `out`, float64
  // [kCalibRows][kCalibCols] in ms. The caller must own the tier, since it claims CPU job sequences, which are the
  // owner's. `tokens` above 1 measures per_token CPU jobs of that many rows, a verify's. device -1 copies with the test
  // backend; `scratch` holds `lanes` experts on that device. Throws on bad
  // arguments, a failed copy or a timeout.
  void calibrate_cpu_split(
      int g,
      int64_t row,
      int64_t device,
      int64_t reps,
      int64_t lanes,
      int64_t tokens,
      uint64_t scratch,
      int64_t scratch_bytes,
      int64_t timeout_ns,
      double* out) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("calibrate_cpu_split");
    const std::string prefix = error_prefix<Layout>() + "calibration: ";
    if (g < 0 || g >= groups()) throw std::runtime_error(prefix + "group " + std::to_string(g) + " does not exist");
    const std::unique_ptr<CpuExpertEngine>& cpu = dist_.group(g).cpu;
    if (cpu == nullptr) throw std::runtime_error(prefix + "CPU experts are not enabled");
    if (copy_engine_ == nullptr) throw std::runtime_error(prefix + "the copy engine is not enabled");
    if (row < 0 || row >= layers_) throw std::runtime_error(prefix + "row out of range");
    if (!cpu->eligible(row))
      throw std::runtime_error(prefix + "row " + std::to_string(row) + " has no registered CPU layer");
    const GroupRow& own = dist_.group(g).rows[row];
    if (own.hi - own.lo < lanes)
      throw std::runtime_error(
          prefix + "it needs " + std::to_string(lanes) + " RAM slots of group " + std::to_string(g) + " in row " +
          std::to_string(row) + ", the group has " + std::to_string(own.hi - own.lo));
    if (reps < 1 || timeout_ns <= 0) throw std::runtime_error(prefix + "reps and the timeout must be positive");
    if (lanes < 1 || lanes > kCalibLanes)
      throw std::runtime_error(prefix + "lanes must be 1.." + std::to_string(kCalibLanes));
    if (tokens < 1 || tokens > cpu->tokens())
      throw std::runtime_error(
          prefix + "tokens " + std::to_string(tokens) + " is not within the CPU rows' 1.." +
          std::to_string(cpu->tokens()));
    BasicCalibrationSetup<Build> s;
    s.cpu = cpu.get();
    s.first_slot = own.lo;
    s.row = row;
    s.entries = copy_engine_->dma_entries(row);
    const int64_t need = lanes * calibration_expert_bytes(s.entries);
    if (need == 0) throw std::runtime_error(prefix + "row " + std::to_string(row) + " copies no bytes");
    if (scratch == 0 || scratch_bytes < need)
      throw std::runtime_error(
          prefix + "the scratch holds " + std::to_string(scratch_bytes) + " bytes, it needs " + std::to_string(need));
    std::unique_ptr<CopyBackend> backend;
    if (device < 0) {
      auto host = std::make_unique<HostCopyBackend>();
      s.host_backend = host.get();
      backend = std::move(host);
    } else {
      backend = std::make_unique<CudaCopyBackend>(static_cast<int>(device), prefix + "copy: ");
    }
    if (const std::string error = backend->init(); !error.empty()) throw std::runtime_error(prefix + error);
    // Release the backend only when idle: after a failure a copy may still be in flight into the scratch.
    struct Shutdown {
      CopyBackend* backend;
      bool idle = false;
      ~Shutdown() {
        backend->shutdown(idle);
      }
    } shutdown{backend.get()};
    s.backend = backend.get();
    s.scratch = scratch;
    s.reps = static_cast<int>(reps);
    s.lanes = static_cast<int>(lanes);
    s.tokens = static_cast<int>(tokens);
    cpu->write_calibration_table(row, tokens);
    s.timeout_ns = timeout_ns;
    calibrate_split(s, out);
    shutdown.idle = true;
  }

  // The split model's startup calibration (split_calibration.h kSplitCell*) on group g: out[i] is the mean ms of
  // cells[i] ({kind, a, b}), over `reps` runs after a warm-up. CPU and DMA runs rotate over `rows` (registered CPU rows)
  // and the group's slots of each; the DMA copies rows[0]'s table into `scratch` (the most DMA lanes of a cell, on
  // `device`; -1: the test backend). A landing cell reads a of the group's own experts of rows[i] into its staging slots
  // through the group's reader (O_DIRECT, so no page cache serves it), the rows rotating. The caller owns the tier and
  // has not armed the copy engine. Throws on bad arguments, a failed copy or read, or a timeout.
  void calibrate_split_model(
      int g,
      const int64_t* rows,
      int64_t row_count,
      const int64_t* cells,
      int64_t cell_count,
      int64_t device,
      int64_t reps,
      uint64_t scratch,
      int64_t scratch_bytes,
      int64_t timeout_ns,
      double* out) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("calibrate_split_model");
    const std::string prefix = error_prefix<Layout>() + "split model calibration: ";
    if (g < 0 || g >= groups()) throw std::runtime_error(prefix + "group " + std::to_string(g) + " does not exist");
    Group& group = dist_.group(g);
    const std::unique_ptr<CpuExpertEngine>& cpu = group.cpu;
    if (cpu == nullptr) throw std::runtime_error(prefix + "CPU experts are not enabled");
    if (copy_engine_ == nullptr) throw std::runtime_error(prefix + "the copy engine is not enabled");
    if (row_count < 1) throw std::runtime_error(prefix + "it needs a row");
    if (reps < 1 || timeout_ns <= 0) throw std::runtime_error(prefix + "reps and the timeout must be positive");
    SplitRotation rot;
    int64_t slots = std::numeric_limits<int64_t>::max();
    for (int64_t i = 0; i < row_count; ++i) {
      const int64_t row = rows[i];
      if (row < 0 || row >= layers_) throw std::runtime_error(prefix + "row out of range");
      if (!cpu->eligible(row))
        throw std::runtime_error(prefix + "row " + std::to_string(row) + " has no registered CPU layer");
      rot.rows.push_back(row);
      rot.lo.push_back(group.rows[row].lo);
      slots = std::min(slots, group.rows[row].hi - group.rows[row].lo);
    }
    rot.slots = slots;  // every row's range is walked as far as the smallest
    SplitRotation copy_rot;
    copy_rot.rows = {rot.rows[0]};
    copy_rot.lo = {rot.lo[0]};
    copy_rot.slots = group.rows[rot.rows[0]].hi - group.rows[rot.rows[0]].lo;
    BasicCalibrationSetup<Build> s;
    s.cpu = cpu.get();
    s.row = rot.rows[0];
    s.entries = copy_engine_->dma_entries(s.row);
    s.reps = static_cast<int>(reps);
    s.tokens = static_cast<int>(cpu->tokens());
    s.timeout_ns = timeout_ns;
    int64_t dma_lanes = 0;
    for (int64_t i = 0; i < cell_count; ++i) {
      const int64_t kind = cells[3 * i], a = cells[3 * i + 1], b = cells[3 * i + 2];
      const bool cpu_cell = kind == kSplitCellHit || kind == kSplitCellMiss;
      if (kind < kSplitCellHit || kind > kSplitCellLanding)
        throw std::runtime_error(prefix + "cell " + std::to_string(i) + " has kind " + std::to_string(kind));
      if (a < 1 || a > Wire::kLanes || a > slots)
        throw std::runtime_error(
            prefix + "cell " + std::to_string(i) + " asks for " + std::to_string(a) + " lanes; the build has " +
            std::to_string(Wire::kLanes) + " and the group's rows " + std::to_string(slots) + " slots");
      if (cpu_cell && (b < 1 || b > s.tokens))
        throw std::runtime_error(
            prefix + "cell " + std::to_string(i) + " routes " + std::to_string(b) + " tokens a lane; the rows hold " +
            std::to_string(s.tokens));
      if (kind == kSplitCellCopy) dma_lanes = std::max(dma_lanes, a);
      if (kind == kSplitCellLanding)
        for (const int64_t row : rot.rows)
          if (a > static_cast<int64_t>(group.rows[row].staging.size()))
            throw std::runtime_error(
                prefix + "cell " + std::to_string(i) + " reads " + std::to_string(a) + " rows; row " +
                std::to_string(row) + " has " + std::to_string(group.rows[row].staging.size()) + " staging slots");
    }
    s.lanes = static_cast<int>(std::max<int64_t>(dma_lanes, 1));
    std::unique_ptr<CopyBackend> backend;
    if (dma_lanes > 0) {
      const int64_t need = dma_lanes * calibration_expert_bytes(s.entries);
      if (need == 0) throw std::runtime_error(prefix + "row " + std::to_string(s.row) + " copies no bytes");
      if (scratch == 0 || scratch_bytes < need)
        throw std::runtime_error(
            prefix + "the scratch holds " + std::to_string(scratch_bytes) + " bytes, it needs " + std::to_string(need));
      if (device < 0) {
        auto host = std::make_unique<HostCopyBackend>();
        s.host_backend = host.get();
        backend = std::move(host);
      } else {
        backend = std::make_unique<CudaCopyBackend>(static_cast<int>(device), prefix + "copy: ");
      }
      if (const std::string error = backend->init(); !error.empty()) throw std::runtime_error(prefix + error);
    }
    // Release the backend only when idle: after a failure a copy may still be in flight into the scratch.
    struct Shutdown {
      CopyBackend* backend;
      bool idle = false;
      ~Shutdown() {
        if (backend != nullptr) backend->shutdown(idle);
      }
    } shutdown{backend.get()};
    s.backend = backend.get();
    s.scratch = scratch;
    fill_join();  // a fill reads through group 0's reader, as a landing cell does
    // The group's own experts (home(e) == g), read in turn so no two runs read one row image.
    const int64_t own_experts = std::max<int64_t>((experts_ - g + Wire::kNodes - 1) / Wire::kNodes, 1);
    int64_t next_expert = 0;
    const auto land = [&](const SplitRotation& r, int64_t m) {
      const int64_t row = r.row();
      std::vector<int32_t> experts;
      std::vector<int64_t> into;
      for (int64_t j = 0; j < m; ++j) {
        experts.push_back(static_cast<int32_t>(g + Wire::kNodes * (next_expert++ % own_experts)));
        into.push_back(group.rows[row].staging[j]);
      }
      std::vector<uint8_t> packed;
      begin_busy(group);  // the watchdog aborts a read that hangs, as it does a demand's
      const int64_t start = now_ns();
      int result = 0;
      try {
        result = group.reader.read(row, experts, into, kBounceRows, [](size_t) { return false; }, nullptr, &packed);
      } catch (...) {
        end_busy(group);
        throw;
      }
      const int64_t ns = now_ns() - start;
      end_busy(group);
      if (result != 1) throw std::runtime_error(prefix + "a read of row " + std::to_string(row) + " failed");
      return ns;
    };
    for (int64_t i = 0; i < cell_count; ++i) {
      const int64_t kind = cells[3 * i];
      const int a = static_cast<int>(cells[3 * i + 1]), b = static_cast<int>(cells[3 * i + 2]);
      switch (kind) {
        case kSplitCellHit:
        case kSplitCellMiss:
          out[i] = split_model_mean_ms(rot, s.reps, [&](const SplitRotation& r) {
            return split_model_cpu_run(s, r, a, kind == kSplitCellHit ? 0 : 1, b);
          });
          break;
        case kSplitCellCopy:
          // The DMA's sources are rows[0]'s copy table: its slots only.
          out[i] = split_model_mean_ms(copy_rot, s.reps, [&](const SplitRotation& r) {
            return split_model_copy_run(s, r, a);
          });
          break;
        default:
          out[i] = split_model_mean_ms(rot, s.reps, [&](const SplitRotation& r) { return land(r, a); });
      }
    }
    shutdown.idle = true;
  }

  // Waits until every job handed to the copy thread has completed or failed; false at the deadline. For the owner
  // (RamThread::pause, once the service parked).
  bool wait_copy_idle_owned(int64_t deadline_ns) {
    return copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
  }

  // The same wait, from any thread (the FFI's copy_engine_idle).
  bool wait_copy_idle(int64_t deadline_ns) {
    return copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
  }

  // The final settle of stop_thread, on the caller once the service joined (RamThread::stop released threaded_ after
  // the join, so this caller owns the tier). A stop that arrives mid-pause can find a prefill fill still running: it is
  // joined first and its epilogue runs here (fill_join). caller_mutex_ orders this against the pausing caller.
  void final_settle() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    fill_join();
  }

  // Test only (InstrBuild): adds one ballast copy of `bytes` to every job's mark, to lengthen the copy.
  void copy_engine_ballast(uint64_t dst, uint64_t src, int64_t bytes)
    requires(Build::kFaults)
  {
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    copy_engine_->set_ballast(dst, src, bytes);
  }

  // The CPU test backend of the copy engine; throws when the engine uses another backend.
  HostCopyBackend& host_copy_backend() {
    auto* backend = copy_engine_ != nullptr ? dynamic_cast<HostCopyBackend*>(copy_engine_->backend()) : nullptr;
    if (backend == nullptr) throw std::runtime_error(error_prefix<Layout>() + "no test copy backend");
    return *backend;
  }

  // Shutdown step (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Shutdown"): called after the device was synchronized while
  // the service still served. No request is in flight, so the service only stops taking new ones.
  void close_admission() {
    admission_closed_.store(true);
  }

  // Reserves every row's staging slots (analysis/dsv41-drive/LEASE_PROTOCOL.md, "Deltas and the bulk delta"): in each
  // group's range the first min(k, slots - 1) FREE slots become kStaging, and the tag-1 delta names them, which the
  // device's first post of the row applies. Runs once, on the tier's owner, before any slot is filled, so no row is
  // evicted and no unmap needs a bulk delta. A group's range with fewer than 2 slots could serve no miss and no hit at
  // once, so it is refused. k == 0 (every streamed layer fully pinned, so no expert can miss) reserves none: the lists
  // stay empty, the device traps on an unforced miss (type_lanes) and the RAM thread fail-stops on any other.
  void reserve_staging(int64_t k) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("reserve_staging");
    if (k < 0 || k > Wire::kLanes)
      throw std::runtime_error(
          error_prefix<Layout>() + "a row has 0.." + std::to_string(Wire::kLanes) + " staging slots");
    for (int64_t row = 0; row < layers_; ++row) {
      const Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        const GroupRow& own = dist_.group(g).rows[row];
        if (own.staging_reserved) throw std::runtime_error(error_prefix<Layout>() + "reserve_staging is once");
        if (own.hi - own.lo < 2)
          throw std::runtime_error(
              error_prefix<Layout>() + "row " + std::to_string(row) + " has too few slots to stage" +
              (dist_.size() > 1 ? " in group " + std::to_string(g) : std::string()));
      }
      for (int64_t slot = 0; slot < tier.capacity; ++slot)
        if (tier.state[slot] != kFree)
          throw std::runtime_error(error_prefix<Layout>() + "reserve_staging is before any slot is filled");
    }
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        GroupRow& own = dist_.group(g).rows[row];
        const int64_t want = std::min(k, own.hi - own.lo - 1);
        for (int64_t slot = own.lo; slot < own.lo + want; ++slot) {
          tier.state[slot] = kStaging;
          own.staging.push_back(static_cast<int32_t>(slot));
        }
        own.staging_reserved = true;
        dist_.seed(row, g, own.staging);
      }
      dist_.publish_seed(lease_, row);
    }
  }

  // Reserves every row's speculative pool (ram_prefetch.h): in each group's range, the first `share` kFree slots after
  // its staging slots become kSpec. Once, on the owner, after reserve_staging and before any slot is filled or the
  // service thread starts; nothing is published, since the device never maps a pool slot. A range left with fewer
  // than 2 slots to serve from is refused.
  void reserve_spec_pool(int64_t share) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("reserve_spec_pool");
    const std::string prefix = error_prefix<Layout>() + "speculative pool: ";
    if (threaded_.load()) throw std::runtime_error(prefix + "reserve it before the service thread starts");
    if (pool_ != nullptr) throw std::runtime_error(prefix + "reserve_spec_pool is once");
    if (share < 1 || share > SpecPool::kMaxShare)
      throw std::runtime_error(prefix + "a row has 1.." + std::to_string(SpecPool::kMaxShare) + " slots per group");
    if (experts_ > 0xFFFF) throw std::runtime_error(prefix + "an entry names experts below 65536");
    for (int64_t row = 0; row < layers_; ++row) {
      const Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        const GroupRow& own = dist_.group(g).rows[row];
        if (!own.staging_reserved) throw std::runtime_error(prefix + "reserve_spec_pool is after reserve_staging");
        if (own.hi - own.lo - static_cast<int64_t>(own.staging.size()) - share < 2)
          throw std::runtime_error(
              prefix + "row " + std::to_string(row) + " has too few slots for a speculative pool of " +
              std::to_string(share) + (dist_.size() > 1 ? " in group " + std::to_string(g) : std::string()));
        for (int64_t slot = own.lo; slot < own.hi; ++slot)
          if (tier.state[slot] != kFree && tier.state[slot] != kStaging)
            throw std::runtime_error(prefix + "reserve_spec_pool is before any slot is filled");
      }
    }
    auto pool = std::make_unique<SpecPool>(layers_, dist_.size(), static_cast<int>(share));
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        const GroupRow& own = dist_.group(g).rows[row];
        int taken = 0;
        for (int64_t slot = own.lo; slot < own.hi && taken < share; ++slot) {
          if (tier.state[slot] != kFree) continue;
          tier.state[slot] = kSpec;
          pool->entry(row, g, taken++).slot = static_cast<int32_t>(slot);
        }
      }
    }
    pool_ = std::move(pool);
  }

  // Pool slots per row and group; 0 without a pool.
  int64_t spec_share() const {
    return pool_ == nullptr ? 0 : pool_->share();
  }

  // Writes {group, slot, state, expert} per pool entry of `row`, group-major; expert -1 for an empty or swapped entry.
  // Any thread: each group's entries under its mutex.
  void spec_pool(int64_t row, int64_t* out) {
    row_capacity(row);
    if (pool_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "no speculative pool (reserve_spec_pool)");
    int64_t k = 0;
    for (int g = 0; g < dist_.size(); ++g) {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      for (int i = 0; i < pool_->share(); ++i) {
        const PoolEntry& entry = pool_->entry(row, g, i);
        const uint32_t word = entry.word.load(std::memory_order_acquire);
        out[k++] = g;
        out[k++] = entry.slot;
        out[k++] = pool_state(word);
        const uint32_t state = pool_state(word);
        out[k++] = state == kPoolEmpty || state == kPoolSwapped ? -1 : pool_expert(word);
      }
    }
  }

  // Enables the RAM prefetch over the reserved pool: per group a speculative thread (RamThread starts it with the
  // service threads) fed by the group's service. The CPU scorer gets every record that staged a CPU input and scores
  // it; the GPU scorer (config.candidates) gets every record of a row with a target and reads its candidate slot. Needs
  // CPU experts on every group. On the owner, before the service thread starts.
  void enable_ram_prefetch(RamPrefetchConfig config) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("enable_ram_prefetch");
    const std::string prefix = error_prefix<Layout>() + "RAM prefetch: ";
    const bool gpu = config.candidates != nullptr;
    if (threaded_.load()) throw std::runtime_error(prefix + "enable it before the service thread starts");
    if (spec_ != nullptr) throw std::runtime_error(prefix + "it is already enabled");
    if (pool_ == nullptr) throw std::runtime_error(prefix + "reserve_spec_pool first");
    if (static_cast<int64_t>(config.target.size()) != layers_ || static_cast<int64_t>(config.gate.size()) != layers_)
      throw std::runtime_error(prefix + "the target table has one entry per streamed row");
    for (int64_t row = 0; row < layers_; ++row) {
      const int32_t target = config.target[row], gate = config.gate[row];
      if ((target < 0) != (gate < 0) || target >= layers_ || target == row || (!gpu && gate >= config.gate_count))
        throw std::runtime_error(
            prefix + "row " + std::to_string(row) + " names target " + std::to_string(target) + " with gate " +
            std::to_string(gate));
    }
    try {
      check_gate_choice(experts_, config.top_k, config.per_token, config.per_layer);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(prefix + error.what());
    }
    if (gpu && config.gate_count != 0)
      throw std::runtime_error(prefix + "the GPU scorer reads the device's gates: pass none");
    if (!config.min_margin.empty()) {
      if (!gpu) throw std::runtime_error(prefix + "min_margin needs the GPU scorer");
      if (static_cast<int64_t>(config.min_margin.size()) != layers_)
        throw std::runtime_error(prefix + "min_margin has one floor per streamed row");
      for (const float floor : config.min_margin)
        if (std::isnan(floor)) throw std::runtime_error(prefix + "a min_margin floor is NaN");
    }
    if (!gpu && (config.gates == nullptr || config.bias == nullptr || config.gate_count < 1))
      throw std::runtime_error(prefix + "it needs at least one gate");
    if (static_cast<int>(config.cores.size()) != groups())
      throw std::runtime_error(prefix + "one core list per NUMA group");
    if (config.idle_deadline_ns < 0) throw std::runtime_error(prefix + "the idle-drive deadline is negative");
    if (config.idle_pieces < 1 || config.idle_pieces > RamPrefetchConfig::kMaxIdlePieces)
      throw std::runtime_error(
          prefix + "idle_pieces must be in [1, " + std::to_string(RamPrefetchConfig::kMaxIdlePieces) + "], got " +
          std::to_string(config.idle_pieces));
    for (int g = 0; g < groups(); ++g)
      if (dist_.group(g).cpu == nullptr)
        throw std::runtime_error(
            prefix + "it needs CPU experts on group " + std::to_string(g) + ", whose records stage the scorer's input");
    const CpuExpertConfig& cpu = dist_.group(0).cpu->config();
    if (!gpu && config.hidden != cpu.hidden)
      throw std::runtime_error(
          prefix + "the gate's hidden size " + std::to_string(config.hidden) + " is not the CPU rows' " +
          std::to_string(cpu.hidden));
    auto spec = std::make_unique<SpecState>();
    spec->x_base = cpu.x_base;
    spec->x_stride = cpu.x_stride;
    spec->x_token_bytes = cpu.x_token_bytes;
    spec->tokens_max = cpu.tokens;
    for (int g = 0; g < groups(); ++g) {
      auto group = std::make_unique<SpecGroup>(std::string(Layout::kName) + "-spec" + std::to_string(g));
      if (!gpu) group->scorer.reserve(spec->tokens_max, config.hidden, experts_);
      group->skip.assign(static_cast<size_t>(experts_), 0);
      group->packed.reserve(1);
      group->cores = config.cores[g];
      group->row_seq = std::make_unique<std::atomic<uint32_t>[]>(static_cast<size_t>(layers_));
      if (config.idle_deadline_ns > 0) {
        // Its own ring over the same tables, counting into the tier's drive load. RowReader reads O_DIRECT only, as
        // the group's reader does.
        const Group& demand = dist_.group(g);
        auto reader = std::make_unique<Source>(Tables(demand.reader.tables()), true);
        reader->set_piece_stream(true);
        reader->set_mirror_map();
        reader->set_drive_load(&drive_load_);
        reader->set_read_kind(kSpecRead);
        reader->set_sq_thread_cpu(demand.sq_thread_cpu);
        if (!reader->open())
          throw std::runtime_error(prefix + "group " + std::to_string(g) + "'s speculative reader did not open");
        group->spec_reader = std::move(reader);
      }
      spec->groups.push_back(std::move(group));
    }
    if (config.promote) {
      // Row r defers its delta when some source targets it; row 0 and untargeted rows publish at once.
      std::vector<uint8_t> deferred(static_cast<size_t>(layers_), 0);
      for (int64_t row = 0; row < layers_; ++row)
        if (config.target[row] > 0) deferred[config.target[row]] = 1;
      dist_.set_deferred(std::move(deferred));
      promote_ = true;
    }
    pool_serve_ = config.pool_serve;
    spec->config = std::move(config);
    spec_ = std::move(spec);
  }

  // Writes every pending deferred delta; the bulk apply kernel skips an unwritten tag. Owner only, once the service
  // threads are parked or joined. A no-op with the flag off.
  void flush_pending() {
    if (!promote_) return;
    for (int64_t row = 0; row < layers_; ++row)
      if (dist_.deferred(row)) publish_pending(dist_.group(0), Request{}, row, 3);
  }

  // The number of eager-path map changes since the last take, {row, expert, slot} each with slot -1 an unmap: the bulk
  // delta that map_bulk_apply writes on the device. Owner only (a paused caller). Joins a running fill first, so a
  // failed fill's unmaps are counted.
  int64_t bulk_delta_count() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("bulk_delta_count");
    fill_join();
    flush_pending();
    return static_cast<int64_t>(bulk_.size());
  }

  // Moves the bulk delta into `out` (3 int32 per change) and clears it; `n` must equal bulk_delta_count().
  void take_bulk_delta(int32_t* out, int64_t n) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("take_bulk_delta");
    if (n != static_cast<int64_t>(bulk_.size()))
      throw std::runtime_error(error_prefix<Layout>() + "take_bulk_delta: the count changed since bulk_delta_count");
    for (int64_t i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c)
        out[3 * i + c] = bulk_[i][c];
    bulk_.clear();
  }

  // The extents the introspection methods below write, for the FFI's exact out-buffer checks, since none of those
  // methods is given a bound. All are fixed at construction, so they take no lock.
  int64_t layers() const {
    return layers_;
  }
  int64_t experts() const {
    return experts_;
  }
  int64_t row_capacity(int64_t row) const {
    if (row < 0 || row >= layers_) {
      throw std::runtime_error(error_prefix<Layout>() + "streamed row " + std::to_string(row) + " is out of range");
    }
    return tiers_[row].capacity;
  }
