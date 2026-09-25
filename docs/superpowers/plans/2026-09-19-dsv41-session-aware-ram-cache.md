# DSV4.1 session-aware RAM cache implementation plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute the checked tasks with separate implementation and review passes. This document is a plan, not authorization to run inference or interrupt production.

**Goal:** improve DSV4.1 throughput at the existing host-memory budget by retaining the experts that prevent the most exposed NVMe reload time for the current session.

**Architecture:** first implement deterministic, decode-only admission/replacement over the existing native pinned slots. Compare exact-count TinyLFU, a sketch estimator, and Window TinyLFU; then add session priors and prediction-aware probation in separately measured stages. The first stage changes retention without speculative reads or moving graph-captured buffers.

**Tech stack:** Python reference/replay and pytest; C++ native metadata policy; existing EXL3 io_uring service, pinned slabs, CUDA graph post/wait/gather, and TVM FFI wrappers.

**Spec and context:** [prediction handoff](../../../DSV4.1%20expert_prediction_handoff.md), [cache background](../../../DSV4.1%20nvme_cache_performance.md), and [DSV41_REFERENCE](../../../DSV41_REFERENCE.md), especially §§16.9 and 18.4–18.6. This plan specifies the cache-policy work that complements the handoff's prediction experiments.

**Snapshot:** 2026-09-19; branch `master`; inspected source `24404a3560`. No implementation, training, replay, or GPU benchmark was run to produce this plan. New files, interfaces, configuration fields, and commands for them below are **proposed**, not existing capabilities.

## Global constraints and current evidence

- Preserve native top-6 routing and healthy-run outputs. A cache policy may reject long-term retention, never an expert required by the current forward.
- Target: 40 layers × 384 routed experts; 5,120-wide router input; 13,315,584 streamed bytes/expert. Keep expert identity as `(layer, expert)`, not expert ID alone.
- Baseline expert RAM: **5,644 rows / 70 GiB**, currently 141–142 rows/layer. Baseline GPU hot set: **888 slots**, inclusive in RAM. Metadata and any scratch remain inside the declared process-memory envelope.
- Preserve **32-forward decode residency updates**. Current paired results favor `c32` over `c0`; typical promotion overhead is about 1–2.5 ms/token. The EXL3 async-promotion flag is now rejected. Do not use the old burst extrapolation as a promised saving.
- Keep original EXL3 shards, drive placement, graph shape, Engram budget, and GPU residency policy fixed during the first policy comparisons.
- First native release is **BS1 decode only, speculative I/O off**. Prefill and eager promotions retain their existing loading behavior. Do not silently apply decode admission to prefill.
- The native protocol allows **8 protected IDs**; decode routes normally contain 6. Use a minimum **8-row logical demand window**. Eager EXL3 gathering supports **64 rows**, so an 8-row window is not sufficient to apply the same policy to arbitrary prefill batches.
- Current graph demand records already include routed GPU hits and update native recency. The missing information is frequency, session/phase, and admission usefulness—not a missing GPU-hit touch.
- Preserve hot/loading/current-consumer protection, map publication ordering, failure signaling, and stable slab/map pointers. A service pause alone does not prove that GPU consumers have completed.
- §16.9's in-sample simulation found that suppressing prefill admission worsened 256-token-prompt results. Selective prefill retention and longer prompts remain experiments.
- Existing `tier_sim.py` skips `graph_step` records, infers phase from token count, and has historical timing constants. It is not yet a validated replay of option C or session-aware admission.
- Shared-host GPU runs require the project's existing production scheduling window. Use the existing local analysis memory limit from [CLAUDE.md](../../../CLAUDE.md): 8 GiB and no swap; use disk-backed artifacts, not a RAM-backed `/tmp` cache. Native/GPU suites need their configured build/runtime environment.

## 1. What to implement and what “TinyLFU” means here

TinyLFU is an **admission decision based on recent frequency**, separate from the replacement policy that nominates a victim. W-TinyLFU adds a recency window before frequency-gated admission to a main cache. A frequency sketch, an exact counter table, and a session prior are separate choices; measure them independently.

Primary references:

- [TinyLFU paper](https://arxiv.org/abs/1512.00727): recent-frequency admission, aging, and the window approach. The paper's Doorkeeper is a distinct estimator component.
- [Caffeine design](https://github.com/ben-manes/caffeine/wiki/Design): an engineering reference for W-TinyLFU and segmented main-cache behavior.
- [Caffeine FrequencySketch](https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/FrequencySketch.java): a compact four-counter, 4-bit implementation with aging. Pin the inspected upstream revision in implementation notes before porting any behavior; do not assume this plan reproduces Caffeine exactly.
- [MoE-Infinity](https://arxiv.org/abs/2401.14361): request-trace-guided expert caching is precedent for the session experiment, not a measured DSV4.1 result.

### Policy variants

| ID | Admission/replacement | Purpose |
|---|---|---|
| P0 `lru` | Existing native LRU | Production reference, unchanged behavior. |
| P1 `window_lru` | 8-row window + LRU main; always accept an eligible window candidate into main | Isolate the capacity/segmentation effect without frequency filtering. |
| P2 `tinylfu_exact` | 8-row window + LRU main; exact aged counts gate admission | Recommended minimal TinyLFU candidate. The window provides required demand staging. |
| P3 `wtinylfu_exact` | Tunable window + segmented LRU main; exact aged counts | Test whether recency and repeated reuse benefit from W-TinyLFU structure. |
| P4 `wtinylfu_exact4` | P3 with collision-free saturating 4-bit counts | Isolate loss from counter saturation. |
| P5 `wtinylfu_sketch4` | Same P3 layout with four-row count-min-style 4-bit estimator | Isolate collisions versus P4 and assess portability/cost. |
| P6 `wtinylfu_session` | Best exact layout plus a causal session prior | Test prompt/session adaptation without extra I/O. |
| P7 `wtinylfu_prefetch` | Frozen winner plus bounded speculative probation | Integrate with the prediction handoff only after retention-only gains are known. |

These are names for the proposed configuration schema, not flags that work today. Do not import a generic cache library into the serving hot path: it cannot own EXL3 slot lifetimes or publication. Implement the small policy locally with no additional runtime dependency.

### Exact estimator: preferred starting point

Allocate `uint16 counts[40][384]`, exactly **30 KiB**, plus one decode-step counter per layer. Count each unique native routed expert once per valid BS1 layer demand, including hits in VRAM and RAM. Never count a prefetch, promotion, startup load, Python lookup, or retry as a new demand observation.

Use saturating increments at 65,535. Before observing the first record of a new aging epoch, halve that layer's counts with integer right shift. Proposed initial epoch is **128 valid decode records per layer**; test 32 and 512 separately. An idle period does not manufacture access events. Carry the aged global history across sessions; reset only session-local statistics at a verified session boundary. At a new process, begin with zeros unless a training-only prior is explicitly configured.

“Exact” means collision-free counts under this declared decay schedule, not an exact sliding-window count. V1 estimators observe decode only; prefill observations are a separate prompt-prior channel. This deliberately differs from a general-purpose TinyLFU that observes every access. Hold the observation population and aging clock fixed when comparing estimators. Exclude graph-capture, warmup, and replay-only executions from logical request demand; extend wrapping protocol sequence numbers into a run-qualified monotonic event identity.

Reference behavior to implement in the new CPU-only policy module:

```python
def observe_decode(counts, ids, *, step, half_life_steps):
    # counts is one layer's mutable 384-entry integer array.
    if step and step % half_life_steps == 0:
        for expert in range(len(counts)):
            counts[expert] >>= 1
    for expert in sorted(set(ids)):
        counts[expert] = min(65535, counts[expert] + 1)


def admit(candidate_score, victim_score):
    return candidate_score > victim_score  # retain incumbent on equal score
```

The `step` argument is zero-based before observation; advance once after each valid layer demand. Apply observation once before the candidate/victim comparison. Observing a batch twice through both `touch` and `assign` is a bug.

### Sketch estimator: an explicit ablation

Use separate tables per layer: 4 rows × width `w` × 4 bits. Sweep widths **64, 128, 256**. Width 128 takes **10 KiB** across 40 layers; a collision-free 4-bit table takes only **7.5 KiB**, so the sketch is not automatically the best memory choice at this expert count.

For each observed expert, increment all four indexed counters, saturating at 15; estimate frequency by their minimum. Use the same decode-step aging schedule as the exact arms, halving all counters at the epoch boundary. This is a controlled estimator comparison, not Caffeine's complete production sampling/reset algorithm.

Define 32-bit unsigned wraparound and the following deterministic hash for Python/C++ parity:

```python
MASK = 0xFFFFFFFF
SEEDS = (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)

def index_for(expert, row, width):
    x = (expert ^ SEEDS[row]) & MASK
    x ^= x >> 16
    x = (x * 0x7FEB352D) & MASK
    x ^= x >> 15
    x = (x * 0x846CA68B) & MASK
    x ^= x >> 16
    return x & (width - 1)
```

Tables are already layer-specific. Reject widths that are not a positive power of two. Test packed-nibble boundary handling explicitly.

Optional P5-D adds an **exact 384-bit first-seen Doorkeeper per layer**: first access sets the bit; subsequent accesses increment the sketch; estimate is `sketch_min + first_seen_bit`; clear bits when counters halve. This avoids a Bloom filter at this small keyspace. Label it an exact-Doorkeeper adaptation, not a reproduction of the paper or Caffeine. Run it only if P4/P5 reveal a reason to investigate first-access pollution.

The Doorkeeper occupies 1,920 bytes across all layers and changes the maximum estimate from 15 to 16. Include an exact counter capped at 16 as a range control before attributing a benefit to first-seen filtering. Caffeine does not use this Doorkeeper; its production aging, adaptive-window, and randomized-admission details are intentionally outside the deterministic v1 comparison.

## 2. Slot-safe admission and W-TinyLFU behavior

Use logical membership metadata; never copy expert payloads just to move between policy segments.

- `H`: all mandatory hot/reserved experts, plus normal physical loading/consumer protection. Mandatory hot rows are not policy victims.
- `W`: demand window, minimum 8 rows. Newly demanded non-hot rows enter here regardless of frequency.
- `B`: main probation segment.
- `R`: main protected segment, used by P3–P7; a real demand hit in B promotes to R. If R exceeds its target, demote its LRU eligible member to B.

Compute soft segment targets from non-hot capacity `C - hot_count`: first W, then main B/R. Initial W=8; main protected target is floor(0.8 × main capacity). Test W=16/32 and protected fractions 0.5/0.8/0.9 after the baseline works. B retains at least one slot when main capacity is nonzero. Soft targets must yield to physical safety and mandatory hot reservations.

### Demand miss sequence

1. Observe the actual demand once; construct the complete protected set for the request.
2. Use a free physical slot if one exists. Place the new row logically in W. When W exceeds its target and main has unused capacity, move its oldest eligible member into main probation without evicting payloads or performing a frequency contest. Do not let a cold cache fill entirely into W merely because free slots exist.
3. If a slot is needed, nominate the oldest **eligible** W member as candidate; exclude hot/reserved/loading/current-protected rows.
4. Nominate an eligible main victim: LRU for P1/P2; B's LRU for P3–P7, demoting an eligible R member if necessary.
5. If candidate wins admission, retain it in main and reuse the main victim's physical slot for the new W arrival. Only membership labels move. If candidate loses, reuse the candidate's slot. A tie retains the incumbent main row.
6. If no main victim is eligible, reuse the eligible W candidate without attempting admission. Do not fail demand because main retention is preferred.
7. Load all required expert bytes, publish using the existing ordering, and let GPU gather complete. **Do not remove a rejected row at request completion:** the GPU has not necessarily consumed it yet. It remains in W until a later proven-safe replacement point.

With advisories disabled and at most 8 unique IDs in `union(need, protect)`, W≥8 preserves staging room for ordinary BS1 demands. Validate the union before counts or membership mutate: the two protocol fields can each hold 8, which does not itself enforce an 8-ID union. Native BS1 posting normally includes needed rows in its protection set. Reject an unsupported oversized union through the existing explicit failure path; never truncate it to fit the window. If safety constraints leave no victim at all, retain the existing explicit failure behavior; do not drop experts or evict an active row to rescue the policy.

Unarmed touch-only requests may update counts, recency, and logical membership but must not unmap/reuse payload storage. Native mutation occurs at the service's existing safe armed-demand path, or within an eager transaction that has completed its consumers. Draining request records and completing producer/GPU work are part of the boundary protocol, not consequences of setting `paused=true`.

Membership-only rebalancing may temporarily exceed a soft target when every prospective member is protected. It never forces a physical eviction to enforce a quota. At hot/reserved transitions and again before selecting an armed-demand victim, replenish W with eligible non-hot main members so the next demand has its guaranteed staging capacity. A depleted W is not a physical no-victim condition if eligible main rows can be relabeled; assert this invariant before issuing reads.

### Prefill and promotion integration

V1 leaves eager prefill admission, ordering, and 64-row chunk support unchanged. Mark policy metadata dirty during legacy eager mutations; at a drained, consumer-safe transition into decode, rebuild logical membership from the actual maps: mandatory H first, newest eligible W next, remaining residents assigned to main. This is metadata reconciliation, not new I/O. Preserve exact demand counts separately from membership.

Mandatory promotion bypasses admission. Reclassify a window resident that becomes hot/reserved into H before it can be selected as a victim; demoted hot rows re-enter main probation. Rebalance soft targets without evicting payloads solely to satisfy a segment size. The existing 64-row inclusive headroom clamp remains in effect; do not quietly replace it with 8 just because decode uses a smaller window.

A later prefill-aware arm must either cap the **inner pinned-admission chunks** to window capacity while preserving the outer EXL3 compute chunk, or provide a separately accounted transient batch region. Measure TTFT and extra rereads. It is not part of v1.

## 3. Session and prompt-conditioned retention

Session-aware retention comes after an unconditioned exact policy has a measured baseline.

Maintain three independent distributions per layer: aged global demand frequency, normalized prompt routing, and current-session decode routing. Count all native selections, including GPU hits, to avoid cache-dependent training bias. Keep future reload value conditional on mandatory residency and the current promotion policy.

For a reproducible initial P6, use an empirical-Bayes-style score in common count units:

```text
prior[e] = 0.5 * normalized_prompt[e] + 0.5 * training_cluster_decode[e]
session_score[e] = session_decode_count[e] + kappa * prior[e]
```

If prompt statistics are absent, use the training global prior. If a cluster is unsupported or out of distribution, also fall back to the training global prior. Use `kappa ∈ {0, 6, 24}` equivalent route observations per layer. Counts and prior refer to selection frequency, not routing-weight magnitude. Decay session counts on a separate 32/128-step sweep so a task change can replace old preferences. Treat this score as a ranking, not a calibrated probability or a claim of optimality.

Fit at most **8 routing-profile clusters** on training sessions only. A profile is a concatenated per-layer normalized prompt histogram; retrieve the nearest centroid using cosine similarity. Associate each centroid with its training sessions' decode distributions. Compare against global-only, prompt-only, and decode-only priors. Prompt semantic labels are reporting strata, not mandatory cache partitions. Do not manufacture “coding” coverage from a finance-only corpus.

Choose the fallback similarity threshold on development sessions, freeze it before calibration/test, and evaluate prompt/decode mismatch explicitly. The first P6 resets session-local counts only at a verified new session; subsequent turns retain them with decay. Cache contents are shared and carried forward—there is no separate weight copy per session.

Require explicit `(session_id, turn_id, phase, forward_id)` context. Do not infer session identity from expert changes or infer phase from `tokens == 1`: a one-token prefill remains prefill. Initial scope is sequential sessions and BS1. Batched/interleaved-session attribution is a separate extension.

For later cost weighting, compare estimated **marginal exposed reload time avoided** over horizons 32/128 output tokens at equal capacity. Use queue/cache replay to avoid multiplying route counts by 10 ms when many uses would require only one reload. Keeping a resident row does not incur a speculative read; fetching an absent row must additionally repay read and pollution costs.

## 4. File and interface map

All new names in this table are proposed. Existing paths are linked for implementation navigation.

| Task area | Files | Responsibility |
|---|---|---|
| CPU policy oracle | Create `python/sglang/srt/layers/moe/expert_ram_policy.py` | Deterministic exact/sketch estimators, segment state, admission decisions; no I/O or GPU imports. |
| Native policy | Create `python/sglang/kernels/jit/csrc/moe/exl3_ram_cache_policy.h`; modify [native service](../../../python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp) | Equivalent CPU metadata policy; invoke it only inside existing safe slot operations. |
| Python/native wiring | Modify [kernel wrapper](../../../python/sglang/kernels/ops/moe/exl3_ram_miss.py), [service](../../../python/sglang/srt/layers/moe/exl3_ram_miss.py) | Configuration, phase/session boundaries, demand observation, telemetry, hot transitions, reconciliation. |
| Eager and residency integration | Modify [expert_stream.py](../../../python/sglang/srt/layers/moe/expert_stream.py), [EXL3 format](../../../python/sglang/srt/layers/moe/exl3_expert_format.py) | Keep prefill legacy; distinguish mandatory loads from actual demand; preserve promotion protection. |
| Configuration | Modify [environ.py](../../../python/sglang/srt/environ.py), [EXL3 requirements](../../../python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py) | Proposed optional `SGLANG_DSV41_RAM_CACHE_POLICY_FILE`; default unset means exact legacy behavior. |
| Replay and reporting | Create `scripts/dsv41/ram_cache_replay.py`, `scripts/dsv41/ram_cache_report.py`; consult [tier_sim.py](../../../scripts/dsv41/tier_sim.py) | Streaming explicit-phase event replay, paired comparisons, manifests; leave historical simulator outputs interpretable. |
| Trace/run harness | Modify [trace_corpus.py](../../../scripts/dsv41/trace_corpus.py), [stream trace](../../../python/sglang/srt/layers/moe/exl3_stream_trace.py); create `scripts/dsv41/ram_cache_ab.py` | Ordered multi-turn sessions, context IDs, policy manifests, graph route events, repeat/order-balanced A/B. |
| Tests | Create `test_expert_ram_policy.py`, `test_expert_ram_policy_replay.py` under `test/registered/unit/layers/moe/`; extend native tier/service/thread tests | Determinism, policy differential checks, protected slots, failure/replay/session correctness. |

Proposed reference API:

```python
@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "lru"
    scope: str = "decode_only"
    window_rows: int = 8
    protected_fraction: float = 0.8
    half_life_steps: int = 128
    sketch_width: int = 128
    session_kappa: float = 0.0
    prefetch_enabled: bool = False

class RamPolicy:
    def observe_decode(self, layer: int, ids: tuple[int, ...], event_id: int) -> None: ...
    def reconcile(self, layer: int, slot_experts: tuple[int, ...],
                  hot: frozenset[int], recency: tuple[int, ...]) -> None: ...
    def choose_slot(self, layer: int, incoming: int,
                    eligible_slots: frozenset[int]) -> int: ...
    def record_fill(self, layer: int, slot: int, expert: int, origin: str) -> None: ...
    def set_context(self, session_id: str, turn_id: int, phase: str) -> None: ...
```

The ellipses above mark interface declarations, not an implementation to land. `eligible_slots` is computed by the storage owner from physical state and the full current protected set; the policy cannot enlarge it. `choose_slot` mutates segment metadata for its selected replacement, with an explicit rollback/reconcile after failed loads. `record_fill` never adds a demand count. `event_id` is a monotonically increasing service-observed demand identity; ring retries and duplicate delivery must not count twice. Reject unseen out-of-order identities in debug/replay mode rather than guessing their order.

Native equivalents use integer session epochs instead of strings. Boundary updates are applied only after preceding records have drained and GPU users completed. V1 can send context through an explicit FFI boundary method; no per-token Python round trip is needed. All publication/maps/bytes remain owned by `RamTier`, not the policy header.

## 5. Implementation tasks and verification

### T0. Establish trace and configuration contracts

**Files:** proposed replay/report modules and trace/run harness above. **Produces:** `events.jsonl`, `sessions.json`, `run.json`, versioned policy JSON; consumed by T1–T7.

- [ ] Define event schema version 1 with run/session/turn/forward/layer/phase; ordered native route IDs/counts; hot/reserved changes; demand/promotion/prefetch origin; load/publication/eviction events; monotonic timestamps where measured; and an explicit missing-event counter.
- [ ] Emit decode route events from the native demand-record consumer without synchronous per-layer GPU reads. Use a bounded telemetry buffer; overflow invalidates the replay run rather than silently discarding accesses. Final throughput runs disable detailed tracing.
- [ ] Record eager prefill counts and chunk order separately. Counts are useful for priors; aggregated prefill records do not reconstruct exact token order, so declare that limitation.
- [ ] Add session/phase callbacks at verified scheduler/forward boundaries. Drain records and complete relevant streams before replacing context. Reject mixed-session batches for this initial mode.
- [ ] Add strict config validation: known mode; W≥8 for graph decode; supported expert/layer counts; positive aging period; valid power-of-two sketch width; fractions in range; no advisory I/O in P0–P6; default unset retains legacy behavior.
- [ ] Capture an 8-session integrity pilot in a scheduled window before the main corpus. Compare native counters, per-layer route counts, physical bytes, replay maps, and session boundaries. A matching event stream is a gate for performance replay.

Proposed unit checks include one-token-prefill phase, duplicate event ID, unknown layer/expert, trace overflow, and context change with an outstanding record. No replay may use `graph_step` aggregates as if they contain individual expert routes.

### T1. Implement estimator and deterministic policy reference

**Files:** new `expert_ram_policy.py`; new policy tests. **Consumes:** config plus explicit demand events. **Produces:** policy decisions and deterministic estimator snapshots.

- [ ] Implement the exact observer and strict admission comparison above; add exact4 and sketch4 as independent estimator types.
- [ ] Implement W/B/R/H metadata and the demand-miss sequence in §2. Free slots precede replacements; ties retain the incumbent; recency ties use slot index for determinism.
- [ ] Implement capture of decision tuples `(event, candidate, victim, candidate_score, victim_score, selected_slot, reason)` for replay/debug only.
- [ ] Implement session reset separately from global decay; no automatic cache flush and no prefetch on context change.
- [ ] Run the focused reference tests, then differential synthetic traces against a deliberately simple full-state reference implementation in the tests.

Required example tests, using the proposed API and estimator classes to implement:

```python
def test_prediction_and_fill_do_not_create_popularity():
    p = make_policy(mode="tinylfu_exact", capacity=12, experts=32)
    p.record_fill(layer=0, slot=0, expert=7, origin="prefetch")
    assert p.frequency(layer=0, expert=7) == 0
    p.observe_decode(layer=0, ids=(7,), event_id=1)
    assert p.frequency(layer=0, expert=7) == 1

def test_aging_and_tie_keep_the_incumbent():
    counts = [0] * 4
    observe_decode(counts, (1,), step=0, half_life_steps=2)
    observe_decode(counts, (1,), step=1, half_life_steps=2)
    observe_decode(counts, (2,), step=2, half_life_steps=2)
    assert counts == [0, 1, 1, 0]
    assert not admit(counts[2], counts[1])
```

Define `make_policy` as a test fixture constructor for one-layer `RamPolicy`, and `frequency(layer, expert)` as a read-only debug accessor. Also assert: main admission rejection still produces a valid demand slot; scan traffic does not evict a repeatedly used main resident in a controlled trace; a changed working set eventually replaces old frequency; exact4 never exceeds 15; hash output matches C++; and aging does not erase neighboring packed nibbles incorrectly.

### T2. Build a faithful replay and choose initial variants

**Files:** new replay/report modules and replay tests. **Consumes:** T0 events and T1 policies. **Produces:** per-policy event results and aggregate reports.

- [ ] Replay P0 first. Validate physical membership and miss counters against the integrity pilot, including touch-only GPU hits, prefill legacy mutations, and promotion loads. Fix mismatches before comparing policies.
- [ ] Preserve chronological session order and cache state within a workload stream. Evaluate a separate explicitly reset cold-start stream; do not mix the two interpretations.
- [ ] Replay P1–P5 on the same demand sequence, capacity, mandatory hot trajectory, and prefill behavior. This fixed-hot replay isolates RAM policy; label it as such.
- [ ] Add a coupled replay that executes the actual residency policy with each RAM policy and counts promotion preparation. Keep its results separate from fixed-hot replay.
- [ ] Replace historical fixed timing constants with versioned measured service distributions when reporting time estimates. Always report miss counts and bytes independently of the latency model.
- [ ] Add a constrained future-access oracle for demand miss count, honoring capacity, mandatory hot sets, and staging. It is not an optimal throughput oracle under arbitrary queues/prefetch.

Required synthetic scenarios: steady hot set plus a one-time scan; alternating disjoint sessions; abrupt within-session task change; flat/unpredictable routing; hot promotion of a W resident; maximum protected demand; and a rejected main admission followed by immediate reuse of its still-resident W row.

### T3. Port the winner's policy to native metadata

**Files:** new native header; existing native service and kernel wrapper; native tier tests. **Consumes:** T1 decision contract. **Produces:** selectable native policy with legacy default.

- [ ] Port exact counts, aging, W/B/R/H state, and deterministic selection first. Add sketch mode only after exact parity passes.
- [ ] Observe actual demand records exactly once in both armed and touch-only paths. Deduplicate `protect`/`need`; do not count promotions through `assign` or prefetch through `serve` as demand.
- [ ] Replace victim nomination inside existing safe mutation points; retain physical eligibility checks and publication/failure ordering outside policy code.
- [ ] Implement hot/reserved membership changes and dirty-state reconciliation after legacy eager operations. Rebuild metadata from surviving mappings after an aborted batch; do not roll physical state back by assumption.
- [ ] Expose policy counters and debug snapshots through FFI for tests. A frequency comparison failure changes residency preference, not request success.
- [ ] Differential-test Python and C++ on identical synthetic traces and the pilot replay; compare selected slots, segment membership, estimates, and physical maps. Include eight simultaneous misses, a disjoint need/protect union larger than eight (rejected before mutation), and a hot promotion that depletes W immediately before a protected mixed-hit/miss request.

Native assertions must cover the existing fixtures' small capacities: either explicitly reject non-LRU configuration when C cannot provide the configured W and mandatory headroom, or create larger fixtures. Do not silently run the algorithm with W=8 over a 3-slot test cache.

### T4. Integrate phase, prefill, promotion, and graph lifecycle

**Files:** service, stream manager, EXL3 format, requirements, and service/graph tests. **Consumes:** T3 safe native policy. **Produces:** BS1 decode mode with reversible configuration.

- [ ] Thread the proposed configuration into service construction before graph capture. Validate unsupported modes before model startup work.
- [ ] Keep prefill on legacy admission; observe its route distribution once per phase through a distinct statistics path. At decode entry, drain preceding records and use the existing completed-consumer boundary to reconcile metadata without payload moves.
- [ ] Ensure reserved promotions are mandatory before the next chunk chooses victims. A row made hot from W must no longer consume the guaranteed non-hot window capacity.
- [ ] Keep slab/map addresses and sizes fixed. Test multiple graph replays, session transitions, promotion boundaries, and eager-prefill-to-graph-decode transitions.
- [ ] Verify rejected main admission loads exact bytes and remains usable through gather. Add delay/failure tests proving that no rejected or expired row is overwritten while consumed.
- [ ] Reject advisory mode in v1; retain current fail-stop behavior under I/O faults and stale publication checks.

Relevant existing regression suites, to run in the configured Linux build environment after implementation:

```bash
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/moe/test_expert_ram_policy.py test/registered/unit/layers/moe/test_expert_ram_policy_replay.py
PYTHONPATH=python python -m pytest -q test/registered/unit/kernels/test_exl3_ram_miss_tier.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py test/registered/unit/kernels/test_exl3_ram_miss_advisory.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py
PYTHONPATH=python python -m pytest -q test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py test/registered/unit/layers/moe/test_expert_hot_cache_publication.py
PYTHONPATH=python python -m pytest -q test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_ram_miss_cuda.py
```

The final two commands require CUDA; the manual suites exercise the EXL3 post/wait/gather route directly. Native CPU suites compile the existing service and need its io_uring/FFI dependencies. Report unavailable prerequisites honestly. These tests were not run while writing this plan. Run full-model healthy token/route/logprob parity against the same fused EXL3 baseline in the scheduled GPU window; existing fused-versus-eager R4 divergence is a separate unresolved issue.

### T5. Add session priors after the retention baseline

**Files:** reference/native policy, context plumbing, replay/report, session profile artifacts. **Consumes:** train-only prompt/decode profiles. **Produces:** P6 and a provenance-checked prior file.

- [ ] Implement global-only, prompt-only, decode-only, and clustered-prior score arms in the common count units specified in §3.
- [ ] Fit centroids and associated decode distributions only on training sessions; validate model hash, 40×384 shape, schema, and source split on load.
- [ ] Select kappa, session aging, and out-of-distribution fallback on development; freeze before locked-test runs.
- [ ] Add mixed-turn and unexpected-topic traces, delayed session switches, and empty/prefix-cached prompt-statistics cases. Missing prefill evidence falls back rather than reusing another session's histogram.
- [ ] Verify this mode causes no read merely because a prior changes. Its first comparison is retention only.

### T6. Evaluate unequal layer budgets as a separate stage

**Files:** replay/report; startup allocation in `expert_stream.py`; configuration validation. **Consumes:** development miss-versus-capacity curves. **Produces:** an optional fixed 40-entry startup allocation.

- [ ] Estimate marginal miss reduction per extra row at each layer under the selected policy and mandatory hot headroom; optimize the total subject to sum=5,644 and each layer's existing capacity/64-row safety constraints.
- [ ] Compare the candidate allocation with equal budgets on held-out sessions. Preserve GPU slot count; reject infeasible allocations before constructing slabs.
- [ ] Apply only at startup. Record how a changed allocation affects allowable GPU residency and include a matched control if the hot trajectory changes.
- [ ] Defer a live global arena or resizing of graph-held pointers to a separate design. This task does not implement dynamic cross-layer reallocation.

### T7. Integrate speculative probation only after P0–P6 are measured

**Files:** predictor RAM bridge from the handoff, native service, policy, telemetry/tests. **Consumes:** a frozen scorer plus safe advisory lifecycle. **Produces:** P7.

- [ ] Reserve speculative occupancy within existing non-hot capacity without consuming the minimum demand window. No reads when the bounded speculative quota or demand queue limit is reached.
- [ ] Give predicted rows an expiry in declared target-forward/layer identity; actual demand converts them to ordinary W/main candidates. Prediction does not increment demand counts.
- [ ] Deduplicate ready/in-flight rows, prioritize demanded reads, and count all completed/discarded/cancelled physical bytes. Current interrupted advisory behavior must be accounted for or redesigned before this arm.
- [ ] Compare scorer-only, prefetch with legacy policy, and prefetch with the chosen retention policy. This separates prediction gains from cache-policy gains.
- [ ] Coordinate leases/generations with actual gather completion. Expiry removes retention preference; it never authorizes reuse of bytes still being read.

## 6. Experiments, parameters, and keep/reject rules

Run narrow comparisons in order. Do not run the Cartesian product of every estimator, window, prior, and workload.

| Experiment | Arms and concrete settings | Rationale | Advance when |
|---|---|---|---|
| E0 integrity | P0 trace/replay, 8-session pilot | Establish that the simulator and event semantics describe option C | Route, membership, miss, and physical-byte accounting reconcile; omissions invalidate a run. |
| E1 staging control | P0 vs P1, W=8, no prefetch | Charge the demand-window design itself | Explain any loss before attributing later gains to TinyLFU. |
| E2 exact admission | P1 vs P2, aging 32/128/512 | Test frequency admission independently of SLRU | A setting reduces development demand+promotion NVMe bytes without correctness loss; no live performance claim yet. |
| E3 W-TinyLFU layout | P2 vs P3; first W=8/16/32 at protected=0.8, then winner at 0.5/0.9 | Test recency/main layout without a full parameter explosion | Better development bytes/estimated exposed wait after accounting for the window's capacity cost. |
| E4 estimator | Best P3 vs P4 vs P5 widths64/128/256; same layout/aging | Separate counter saturation from collisions and runtime overhead | Prefer exact unless sketch improves measured cost without materially worse cache decisions. Optional Doorkeeper is an additional arm. |
| E5 session | Winner vs P6 global/prompt/decode/cluster priors; kappa0/6/24 | Test actual prompt-type adaptation and negative transfer | Improvement survives unseen sessions and task switches; OOD fallback prevents persistent regressions. |
| E6 prefill | Legacy admission first; normalized-prior ablation; selective prefill only after separate chunk-safety work | Preserve useful warm-up while testing long prompts | End-to-end request time and TTFT justify changes; do not conclude from decode rate alone. |
| E7 layer budget | Equal vs one frozen development-derived allocation | Test marginal value of RAM per layer | Held-out gain with fixed total rows and valid inclusive headroom. |
| E8 native overhead | P0 vs count-only vs selected native policy, detailed trace off | Separate statistics CPU cost from replacement benefit | Policy bookkeeping does not erase predicted savings or add synchronization to the layer path. |
| E9 live serving | P0 vs selected exact/W variant; add P6 only if offline gate passed | Establish real throughput benefit | Pass the predeclared live criteria below. |
| E10 prediction | No prediction; scorer-only; frozen prefetch+P0; same prefetch+winner | Test cache/prefetch interaction | Positive net serving gain including extra disk traffic and pollution. |

### Dataset and workload design

Create `sessions.json` with **96 distinct sessions**: 32 train, 16 development, 16 calibration, and 32 locked test, grouped by whole source session and near-duplicate prefix. Use a fixed recorded seed for assignment. If a desired prompt class is absent from the existing corpus, add representative sessions before claiming results for that class; otherwise label the experiment's narrower coverage.

For trace acquisition, stage work: 8-session integrity pilot first; then collect the training/development/calibration traces within scheduled windows; acquire/evaluate locked-test traces only after the policy and reporting choices freeze. Capture routing/count/state events, not full expert tensors or 5,120-wide hidden states, for these cache-only experiments. Stop on the configured disk/memory cap and record incomplete runs explicitly.

Report strata for 256-token, 1,024-token, and 4,096-token prompts where the original content supports those lengths; do not pad with repeated tokens and call it representative. Use 128 generated tokens for the controlled throughput comparison; additionally retain naturally short completions as a separate end-to-end workload. Use the existing benchmark's fixed-output-length convention consistently across paired arms.

Exercise these chronological workloads: single-topic sessions; several turns of one session; within-session task changes; alternating unrelated sessions; long prefill followed by short decode; and warm consecutive requests. Declare whether caches reset before each session or persist through a stream. Use both cold-isolated and warm-chronological evaluations; never transfer warmed policy state from one experimental arm into another.

Start live screening on 8 development sessions. For the locked live comparison, run P0 and the frozen winner on the same 32 test sessions with two order-balanced repetitions, preserving identical workload order within each warm stream. Use session/stream-block resampling for confidence intervals, not independent-token resampling. The runner must report both mean per-session rate and total output tokens divided by total measured decode wall time.

### Predeclared live acceptance criteria

These are proposed engineering thresholds to freeze before the locked test, not results:

1. Healthy native routes and deterministic output tokens match P0 on the same fused numerical path; any unexplained mismatch blocks acceptance.
2. The paired 95% confidence interval for decode wall-time reduction is above zero, with at least **3% point-estimate improvement**. Smaller reliable gains may be documented, but do not silently lower this promotion criterion after seeing test results.
3. No more than **5% regression** in p95 inter-token latency or TTFT in any adequately sampled declared workload stratum. Report p99, worst session, and sparse-stratum uncertainty; insufficient tail evidence is inconclusive rather than a pass.
4. For retention-only P1–P6, total expert NVMe bytes per output token including promotion/reread traffic must not increase by more than **2%**. Also report prefill bytes and total request time; a fixed short output makes TTFT material.
5. Fixed 5,644 RAM rows and 888 GPU-hot slots, no new synchronous per-layer GPU-to-CPU readback, and measured policy metadata/CPU cost within the recorded memory budget.
6. If tests or runtime show an unsafe slot lifetime, stuck request, missing event attribution, or unbounded queue, performance results do not qualify. Preserve and report negative sessions.

For P7, freeze a separate extra-I/O ceiling on calibration data because speculative traffic is intentional. Do not apply the retention-only 2% limit blindly; report the chosen ceiling and the net latency tradeoff before opening the locked test.

## 7. Proposed configuration and reproducible outputs

Example policy file for implementation and tests; **not usable until T0/T3/T4 implement its loader**:

```json
{
  "schema_version": 1,
  "mode": "wtinylfu_exact",
  "scope": "decode_only",
  "window_rows": 8,
  "protected_fraction": 0.8,
  "half_life_steps": 128,
  "sketch_width": 128,
  "session_kappa": 0.0,
  "prefetch_enabled": false
}
```

Proposed replay CLI to implement in T2, shown with deliberate file arguments rather than invented currently available flags:

```bash
PYTHONPATH=python python scripts/dsv41/ram_cache_replay.py --events artifacts/dsv41-ram-cache/pilot/events.jsonl --policy artifacts/dsv41-ram-cache/config/wtinylfu-exact.json --ram-rows 5644 --gpu-hot-rows 888 --out artifacts/dsv41-ram-cache/replay/wtinylfu-exact
```

The new CLI must validate schema, capacities, trace completeness, phase/session identities, and policy prerequisites before replay. `run.json` records every argument, source revision, event hash, model hash, driver/device/drive conditions, workload order, seed, cache initial state, and elapsed/resource limits. Output `summary.json`, `per_session.jsonl`, `per_layer.jsonl`, and bounded decision samples. Detailed events and raw outcomes remain available for an independent reviewer.

The A/B runner creates isolated server processes per arm/stream, carries state only where the manifest requests it, uses the existing validated serving launch configuration, and records effective configuration. Do not copy an old eager or Qwen launch and assume it describes option C.

## 8. Delivery order and handoff

**First milestone:** T0–T2, CPU reference and validated replay, with no serving policy change. Return the P0/P1/P2 opportunity estimate and constrained miss-count oracle gap.

**Second milestone:** T3–T4 and E8/E9 for the simplest qualifying exact policy. This can finish before session clustering, unequal layer allocation, or prefetch work.

**Third milestone:** T5 and E5 for prompt/session adaptation. Preserve a neutral fallback if workload type is uncertain.

**Optional milestones:** T6 layer budgets, selective prefill admission, sketch/Doorkeeper refinements if justified, and T7 prediction integration. Each has its own measured decision; none is required to claim the first milestone complete.

Leave the new configuration unset to roll back to the current policy. Independent review must inspect both the numerical policy and the physical slot lifecycle; Python policy tests alone cannot approve native publication or graph safety. The final implementation report lists exactly which variants ran, what remained proposed, all test/benchmark artifacts, and whether the frozen acceptance criteria passed.
