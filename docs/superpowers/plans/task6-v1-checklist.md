# Task 6, V1 (two-phase): implementation checklist

**Companion to** `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, Task 6.
That section's checklist is **not this one** and must not be followed for this work.

Every line number below is against **`f3f66fb2d1`** on `dsv41`. They are given so a
reader can find the code, not so a reader can trust them after a rebase; re-check before
editing.

---

## 0. Which mechanism this specifies, and which two it does not

This checklist specifies **V1, the two-phase mechanism: publish each hit lane's
`RowResult` inside `serve()`'s reservation critical section so the device can copy the
resident rows while the misses are still being read, then copy the rest as today.** It is
the chosen mechanism. It does **not** specify **V2 at fixed lane order** — the variant the
checklist and pseudo-code still standing in the plan's Task 6 describe, measured at
**-15.67 ms/step against two-phase** and decisively rejected; if your work is growing one
stage per lane, you are building that one. It also does **not** specify **V2 at best
order** — the marginal variant (gross +4.93 ms, net +2.7 to +3.6 ms, 1.1-1.4% against a
1.5% resolution bar), which is expected-REJECTED but not settled and turns on `g` and `c`,
both unmeasured. V1 has **two** stages, always, whatever `k` is. If a step below would
give you more than two, it is wrong and you should stop rather than generalise it.

---

## 1. State of the tree, verified rather than inherited

Re-verified 2026-09-21 by reading the source at `f3f66fb2d1`. Items 1, 3, 4 and 5 below
are places where the plan or the brief that commissioned this checklist is **wrong or
imprecise** about the code; items 2, 6, 7 and 8 are confirmations that change what a step
must do. Each one changes a checklist item, so they are stated here rather than in a
footnote.

1. **`grant_lanes_locked` is not "the all-hit path".** It has exactly one caller,
   `serve()` at `exl3_ram_miss_host.cpp:2614`, inside `if (ok && lease_mode_ &&
   !advisory)`. It runs for **every** successful lease-mode demand, hit or mixed. The
   `kReady` check at `:2091` is a post-condition that miss lanes satisfy because the
   publish block at `:2591-2610` has just set `tier.state[slots[i]] = kReady` for them.
   The plan's conclusion is nonetheless right: today **no lane's `RowResult` is published
   before `read()` returns**. V1's service-side delta is therefore not "add a path" but
   **"split the one grant in two"**.

2. **`serve()` drops `mutex_` between reservation and `read()`**, at `:2546`. Confirmed.
   The plan's insistence that the hit lease be taken in the *same* hold as the reservation
   is correct and is the reason step S2 is where it is.

3. **The plan's V2 landmine is stated against a throw that does not exist.** Task 6 says
   "Task 5 makes releasing a leased slot throw, so the cleanup must skip published lanes
   or the service thread throws on any mid-request failure." The throw is in the *public*
   `release(row, slot)` API (`:1982`), which Python pin/unpin uses. `serve()`'s cleanup
   calls `release_locked` (`:2439`), which has **no lease check at all**: it publishes map
   `-1` and sets `kFree`. `take_slot_locked`'s free-slot scan (`:2408-2409`) then returns
   that slot with no lease check either. So the landmine's failure is **silent wrong
   bytes plus lease-count corruption** (the later `release_lease_locked` decrements
   `tier.leases[]` on a slot that now holds a different expert), not a loud throw. Nothing
   will notice.

4. **V1 is not exposed to it — for a narrower and checkable reason than the plan gives.**
   The plan says "V1 does not hit this: its hit lanes are already `kReady`." The operative
   fact is different: `serve()`'s two `release_locked` sites (`:2541`, `:2604`) both
   iterate `slots`, which only ever holds **newly taken** slots for experts in `missing`.
   A hit lane's expert is by definition not in `missing`, so its slot is never in `slots`.
   Additionally `:2490` pushes every `lane_expert` into `wanted`, and `take_slot_locked`
   is called with `fallback = false`, so a `wanted` expert is never its own request's
   victim. Both facts are load-bearing for V1; step S6 turns them into assertions.

5. **V1 *is* exposed to a failure the plan does not name.** Today `grant_lanes_locked`
   runs only when `ok`, so a failed request grants nothing. Under V1 the hit leases are
   granted **before** `read()`, so a read error, a fault or a cancel leaves them
   outstanding. The host already copes — `retire_leases` (`:2126`) resolves acknowledgement
   and terminal mask per lane independently — but the device publishes a **whole-request**
   mask today (`exl3_ram_miss.cuh:513`, `(1u << named) - 1u`). Making that mask partial is
   a device change; `retire_leases` needs no edit for it. One small host edit *is* needed,
   for a different reason: see S4.

6. **`state[kSticky]` is never cleared.** `exl3_ram_miss.cuh` sets it at `:217`, `:358`
   and `:516`; there is no clear site in the kernels, and nothing in
   `ops/moe/exl3_ram_miss.py` re-zeroes the state tensor after `:790`. It is a
   process-lifetime fail-stop latch, not a per-request flag. Do not use it as `req_failed`.

7. **V1 needs no change to `RowReader::read` and no `pack_one` callback.** That hook is
   V2's, for publishing miss rows at their own pack. V1 publishes misses exactly where
   they are published today. If you find yourself editing `read()`, you have drifted into
   V2.

8. **`sim_ack` and `hold_until_ack_lane` do not exist** anywhere in `python/` or `test/`.
   Both are `[P]` proposals in `PER_ROW_TRANSFER.md`. The tests below are specified so
   that the host-side ones need neither.

---

## 2. Service-side checklist (host C++, `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`)

- [ ] **S1. Split `grant_lanes_locked` (`:2079`) into an entry-opener and a per-group
      granter.** Today it does six things in one call: reject a still-active ring entry (`:2083`),
      check every lane resident (`:2091`), reinitialise the `Outstanding` entry, write payloads and
      take leases, `_mm_sfence()`, then store the ready words. V1 needs the entry opened
      **once** with the full lane count and every lane marked *ungranted*, and a granter
      callable **twice** over a subset of lanes. Each call keeps its own
      `_mm_sfence()` between its payload writes and its ready stores — do **not** hoist a
      single fence to cover both groups; the hit group's ready words must be visible while
      the miss group's payloads do not yet exist.

- [ ] **S2. Grant and publish the hit lanes inside `serve()`'s reservation `mutex_` hold**
      (the scope opened at `:2497`), **after** the `take_slot_locked` loop has completed
      with `ok` still true, and **before** the scope closes at `:2546`. A lane is a *hit*
      iff its `request.lane_experts[lane]` is not in `missing`.
      Two orderings are wrong and must be avoided by name:
      - *Before the take loop*: the `!ok` bail at `:2539` releases `slots` and returns with
        no lease unwind, and the deferral branch at `:2519` sets `ok = false` for a request
        that will be **retried under the same seq**. Either leaves leases outstanding for a
        request that never ran.
      - *After the mutex drop at `:2546`*: that is the eviction window this entire task
        exists to close, and a grant there buys nothing.

- [ ] **S3. Narrow the existing grant at `:2614` to the miss lanes only**, and make it
      reuse the entry S2 opened. It must not execute `entry = Outstanding()` (`:2094`),
      which would erase the hit leases S2 recorded. Leave it where it is in `serve()`.

- [ ] **S4. Teach `retire_leases` that an ungranted lane is still open.** The `open` loop
      at `:2156-2157` counts only `state == 1` as open. Between S2 and S3 a miss lane sits
      in the ungranted state, so `open` would be false and `entry.active` would clear while
      a grant is still pending — after which `grant_lane_locked`'s `entry.active` guard
      (`:2083`) no longer protects the ring slot. This is the only host change outside the
      grant itself.

- [ ] **S5. Enumerate and *deliberately do nothing* on the failure paths.** Under V1 these
      now hold hit leases where today they hold none: `fail_reads` (`:2564-2568`),
      `reader_.read` returning 0, `cancelled` (`result == -1`; reachable only on the
      advisory path today — record that, because it becomes reachable for demands if the
      cancel predicate ever widens), and S3's own grant failing. In every one the hit
      leases **stay outstanding** and the request answers with a non-`kServed` status. They
      are retired by the device's stage-1 acknowledgement or voided by its terminal (D4).
      Do **not** release them on the host: that is exactly the silent-corruption path of
      §1.3.

- [ ] **S6. Make §1.4's two facts assertions rather than arguments.** Add a debug
      assertion at `:2541` and `:2604` that the slot being released is unleased. Do **not**
      add a lease check to `release_locked` itself and do **not** route the cleanup through
      `release` — that is a Task 5 decision about the V2 landmine, not V1's to make
      (open question O2).

- [ ] **S7. Separate the counters.** `counters_[kLeasesGranted]` is incremented once per
      request at `:2119`. Split it, or add a `hit_leases_granted` counter, so that an arm
      or a test can show the hit group was granted **before** `read()` at all. Without a
      counter that distinguishes the groups, a build that publishes nothing early passes
      every timing test below by falling through to the batched path.

---

## 3. Device-side checklist (CUDA, `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`)

Kept separate because the service-side delta alone delivers nothing: the device still
polls `kDemandDone` and would not observe the early words. The plan's item 3 predates the
`RowResult` words existing, which is why this half is easy to skip.

Target chain per streamed layer, replacing today's post → wait → copy → ack:

```text
post  ->  W1  ->  C1  ->  A1  ->  W2  ->  C2  ->  A2  ->  F  ->  fused_moe
          (hit stage)              (the rest)          (finalize; sole writer of keep)
```

- [ ] **D1. New `exl3_ram_miss_lease_hit_wait_kernel` (stage 1).** It **must not poll
      `kDemandDone`** — that is the whole point of the task. It polls each planned lane's
      `RowResult.ready` and **compacts** the ready-and-valid lanes into its own `rows_1[]`,
      `slots_1[]` and `lane_ctx_1[]`, committing the count to `go_1[0]` as its single last
      store (fail-closed, exactly as `:507` and `:513` do today).
      - Reuse the existing validity test at `:468-479` (generation, `kLeaseTagReady`,
        seqlock re-read, expert identity, `host_slot < capacity`). Do not write a second
        copy of it.
      - **The one difference that a copy-paste will get wrong:** in stage 1 a lane that is
        *not* ready is **not a failure**. It is a miss lane and belongs to stage 2. The
        `misses > 0 -> ok = false; reason = kLeaseReasonIdentity` branch at `:486-491`
        must **not** apply here. A lane that is ready but *invalid* (wrong expert, out of
        range, torn seqlock) is still a hard failure. Keep those two cases apart.
      - **Bound the poll.** Stage 1 polls for a bounded number of iterations or until
        `kDemandDone` is reached, whichever comes first, then commits what it has. Without
        a bound, an all-miss request pays the read wait twice (test T10).
      - *Contiguity.* The copy kernel takes base pointers plus a count, so stage 1's lanes
        must be contiguous — hence compaction inside the stage kernel. The alternative,
        `ord`/`h` written by the post kernel (`PER_ROW_TRANSFER.md` §4.2), is **not**
        specified here; see open question O1.

- [ ] **D2. Change `exl3_ram_miss_lease_wait_kernel` (`:382`) into stage 2.** It cannot be
      reused unmodified.
      - It must skip the lanes stage 1 committed: it needs `go_1` and stage 1's compaction
        map as inputs, and must build `rows_2[]`/`slots_2[]` over the complement.
      - `state[kPending] = 0` at `:458` must move to whichever stage runs last, or both
        stages must read `kPending` before either clears it.
      - **It must stop writing `keep`.** `:504` writes `1.0f` and `:518` writes `0.0f`.
        With two stages, stage 2's `keep = 1` overwrites a `keep = 0` that stage 1's
        acknowledgement kernel already wrote on a violation. Only `F` writes `keep`
        (`PER_ROW_TRANSFER.md` DECIDE 4). Test T6 exists for exactly this.
      - `ram_miss[0] += misses` is accumulated at `:505` and `:519`. Pick one owner across
        the two stages or it double counts.

- [ ] **D3. Run `exl3_ram_miss_lease_ack_kernel` (`:525`) once per stage**, each over its
      own `go_s` and `lane_ctx_s`. Its `keep[0] = 0.0f` at `:554` must become a `violated`
      flag that `F` reads; its `raise_fatal` may stay.

- [ ] **D4. New finalize kernel `F`**, stream-ordered after every `C_s` and `A_s` and
      before the fused MoE. It is the **only** writer of `keep`. Success is
      `no stage failed && go_1 + go_2 == planned_count && page fatal == 0 && no VIOLATED
      ack`. On failure it publishes a **partial** terminal mask: the lanes acknowledged by
      no stage. Remove the whole-request mask at `:513`. Leaving it would void the hit
      lanes stage 1 already acknowledged; `retire_leases`'s per-lane state machine
      (`:2150-2154`) keeps that from corrupting anything, but it records it only as
      `kLeaseDoubleSignal`, so the wrong mask would otherwise be invisible.

- [ ] **D5. One request deadline, shared by both stages.** The post kernel computes
      `deadline = global_ns() + timeout_ns` once into a device word; each stage compares
      against that absolute value, not against its own start. Otherwise a two-stage request
      can run to 2x `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`. `state` is `int32[]`, so a 64-bit
      deadline needs two words or a new tensor. Any new `SGLANG_*` flag follows
      `env-var-conventions`.

- [ ] **D6. Add a per-request failure flag; do not press `state[kSticky]` into service.**
      Stage 2 and `F` need "an earlier stage of *this request* failed". `kSticky` is never
      cleared (§1.6), so using it would fail every subsequent request in the process.

- [ ] **D7. Rebuild the chain in `Exl3RamMissRowBackend.post`**
      (`python/sglang/srt/layers/moe/exl3_ram_miss.py:306`). Python runs at capture only;
      every buffer preallocated, every view sliced at capture. `Exl3RamMissDevice`
      (`python/sglang/kernels/ops/moe/exl3_ram_miss.py:798`) allocates **one** `go_count`
      and **one** `lane_ctx`; it needs one per stage, and `_owned_tensors`
      (`srt/layers/moe/exl3_ram_miss.py:712`) must list the new ones or they are freed out
      from under the captured graph.

- [ ] **D8. Register the new kernel names** in the `names` tuple at
      `python/sglang/kernels/ops/moe/exl3_ram_miss.py:740`. Omitting one fails at JIT load
      with an error that will not say why.

---

## 4. What must be tested, and the mutant that proves each test is not vacuous

A passing test proves nothing here until it has been seen to fail. **Derive each mutant
from the sentence the test claims, not from the function the test calls** — the plan's
SAFETY passage records a case where the obvious mutant left the property's own test green
because LRU recency selected the same victim either way. Where a claim has no falsifying
mutant, that is said out loud rather than papered over.

| # | Claim | Test | Mutant that must turn it red |
|---|---|---|---|
| **T1** | Hit lanes are published **before `read()` returns**. | Host only. `inject(delay_s = 2.0)`; a request with at least one resident and one missing expert. From the test thread, read the lease block's `RowResult[idx][lane].ready` directly (`exl3_lease_block.ROW_RESULT`, `ROW_RESULT_FIELDS`) while the service is inside `read()`. Assert the hit lane's word carries `READY_TAG` and the request's generation, the miss lane's word is still zero, and `kDemandDone` has not reached the seq. | Move the S2 grant back to `:2614` (today's code). Must go red **on the ready word**, not on a timeout. A test that only asserts "both words are eventually set" is vacuous and passes unmodified `f3f66fb2d1`. |
| **T1b** | The payload lands before the ready word. | Same setup; the seqlock re-read of `:477`. | Write the hit lane's ready word before its payload (drop S1's per-group fence). If this does not go red, the ordering claim is untested and the fence is decoration. |
| **T2** | The hit lease is taken in the **same `mutex_` hold** as the reservation. | `slot_info(row)` shows `leases > 0` on the hit slot at the same moment T1 observes the ready word. | **None exists.** With one service thread there is no second actor to interleave, so taking the lease in a *second* `mutex_` hold immediately after the first leaves this test green. This claim rests on a code reading, and the checklist says so rather than claiming coverage it does not have. See O3. |
| **T3** | A hit lane's slot is never its own request's victim. | Tier at capacity exactly `k`, every slot resident, the hit expert among them, the reservation needing a victim. | **Two mutants, both required.** (a) Force `listed(protect, expert)` false in `take_slot_locked` — per the plan's SAFETY passage this may stay green through recency, and that is the point of running it. (b) Delete `:2490`, the `lane_experts -> wanted` loop; a lane expert that is not routed then becomes a legal victim of the request leasing it. (b) is the one matched to the claim. |
| **T4** | A failed request voids the hit leases and **releases no slot**. | `inject(fail_reads = True)` on a mixed request whose hit lane was granted at S2. Assert: right after the request answers, `slot_info` shows the hit slot still `kReady` with `leases == 1`; the device terminal names the hit lane; after it, `leases_voided == 1` and `leases == 0`. | Add `release_locked` for the hit slot to the `!ok` path. Must go red on `leases`; a **second** assertion must catch the consequence — issue a request for a different expert and assert it does not land on that slot. One assertion alone would pass on a build that corrupts the count without reusing the slot. |
| **T5** | The partial terminal mask names **only** unacknowledged lanes. | GPU. Stage 1 copies and acknowledges; stage 2 fails on an injected read failure. Assert `skipped_mask` has the hit lane's bit **clear** and the miss lane's set, and `kLeaseDoubleSignal == 0`. | Restore the whole-request mask `(1u << named) - 1u` at `:513`. Must go red on the mask bit **and** raise `kLeaseDoubleSignal`. Two independent detectors for one mutant is what makes the mask claim non-vacuous. |
| **T6** | `keep` has exactly one writer. | GPU. Force a VIOLATED acknowledgement in stage 1 (bump the slot generation between grant and acknowledgement) and let stage 2 succeed. Assert `keep == 0` and no fused output. | Restore `keep[0] = 1.0f` in the stage-2 wait. Must go red. This is the concrete bug D2 exists to prevent and it is invisible without this test. |
| **T7** | One request deadline, not one per stage. | Delay both stages to 0.9x the timeout; the request must fail at ~1x, not ~1.8x. | Give each stage its own `start + timeout_ns`. |
| **T8** | The captured graph is the linear chain D7 builds. | Enumerate kernel-node dependencies with `cudaGraphGetEdges`; assert post → W1 → C1 → A1 → W2 → C2 → A2 → F → fused. | Capture A1 on a second stream. (`PER_ROW_TRANSFER.md` §5.2 item 1 requires this be checked, not assumed.) |
| **T9** | Output parity. | Destination rows byte-equal and fused output bitwise equal against **A1** (Task 5 lease-mode batched) on fixed routes and seeds, eager and graph. | Swap two lanes' destination slots in stage 1's compaction. This is the failure mode compaction introduces and `ord` would not — it is the price of O1's recommendation and must be paid in a test. |
| **T10** | An all-miss request does not pay the read wait twice. | Every lane misses. Assert stage 1 commits `go_1 == 0` promptly, with a wall-clock upper bound. | Remove D1's poll bound and let stage 1 spin to the deadline. |
| **T11** | The batched path survives as stage 1 covering every lane. | All-hit request: assert `go_1 == k`, `go_2 == 0`, the empty stage acknowledges nothing, and a poisoned source slab is not read by stage 2. | Make an empty stage's acknowledgement kernel acknowledge lane 0. Must go red on `leases_acked`. |

Existing files to extend rather than duplicate:
`test/registered/unit/kernels/test_exl3_ram_miss_leases.py`,
`test_exl3_ram_miss_lease_service.py`, `test_exl3_ram_miss_lease_thread.py` (T1-T4, T7 host
half); `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`,
`test_exl3_ram_miss_graph_gpu.py` (T5, T6, T8, T9, T11).

---

## 5. Reporting requirement and baseline

**Baseline is A1 = Task 5 lease-mode batched**, not today's unleased path. Lease mode arms
every `count > 0` record, so all 40 layers pay a service round trip that A0 does not.

**Report all three differences, each with an interval: A1 v A0, A2 v A1, A2 v A0**
(A0 = today's unleased path, A1 = Task 5 lease-mode batched, A2 = this mechanism). A2 v A0
alone credits V1 with the cost it inherits from Task 5; A2 v A1 alone hides a possible loss
inside Task 5. Neither is "the Task 6 result" on its own.

**Net out OPEN 11: 0.68 ms/step at 40 layers, 17 us per armed layer**, re-measured
2026-09-21 on an exclusively held card (`analysis/dsv41-drive/open11/results.md`). That is
**~61% of the 1.114 ms `G*`** Task 6 is trying to win. **This supersedes the ~8 us /
~0.32 ms / "28-30% of `G*`" figures still carried in `LEASE_PROTOCOL.md` §17.2 (~line
1462) and OPEN 11 (~line 1874)**, which were not updated when `open11/results.md` was. A
reader who reaches the baseline requirement through §17.2 gets the wrong number.

**Denominator: a DSV4.1 EXL3 in-graph decode step is ~360 ms** (2.781 tok/s,
`DSV41_REFERENCE.md:4`). **Do not use 66.8 ms/token** — that is a Qwen3.8 NVFP4 figure
from `analysis/step-tail`, on a prefetch server with 48 layers per step. The correction is
already recorded at `open11/results.md:168`.

**Preconditions on every arm, stated in the arm's own record:** promotions off, prefetch
puller off, eager (speculative) gather off — not prefill's gather, which is not on this
path. They share the Gen3 link, and contention on it subtracts from the hiding in the
direction that flatters a null result. A combined Task 6 + Task 8 arm is a separate,
labelled experiment.

**Pre-registered falsification (`PER_ROW_TRANSFER.md` §5.6), fixed before measuring:** V1
is falsified if the untraced tok/s ratio V1 / A1 has a 95% interval containing 1.0, or if
the point estimate is below 3%. The modelled prediction is 20.8 to 35.7 ms/step, 8% to 14%
of the step — **modelled, not measured**, a ceiling by construction (35.74 = 33.9 hit lanes
x `c` = every hit copy perfectly hidden), and nothing in it models a hit copy that fails to
hide.

**The Gate's clause 3 still applies.** V1 reads no `slot_map` before `demand_done` —
correctness comes from the per-lane `RowResult`, which is why it is exempt in fact. State
that exemption explicitly in the acceptance record rather than leaving it inferred, since
the clause is a refusal.

---

## 6. Open questions — stated as open, not guessed

- **O1. Compaction in the stage kernel, or `ord`/`h` from the post kernel?** This
  checklist specifies compaction (D1), because it keeps the hint out of the device
  protocol entirely and `ord` buys only poll-read savings. **The owner should rule**,
  because the two documents disagree about whose mechanism `ord` is:
  `PER_ROW_TRANSFER.md` §4.1's table gives **V1** the stage ranges `[0, h)` / `[h, k)`
  (which *is* `ord` ordering) and §3.2 closes with a paragraph headed "Consequence for
  **V1's** order array `ord`", while the plan's Gate text reads as though `ord` were V2's
  alone. The plan's warning that "there is no `ord` array in [the checklist and
  pseudo-code]" is a criticism of the **V2** checklist, not a prohibition on V1 using one.
- **O2. Should `release_locked` grow a lease check?** It would make the V2 landmine loud
  instead of silent (§1.3). V1 does not need it; S6's assertion is the cheap half. Whether
  to harden `release_locked` itself is a Task 5 decision and is not taken here.
- **O3. The same-`mutex_`-hold claim (T2) has no falsifying mutant** with one service
  thread. Whether that is acceptable, or whether a test-only second thread is worth
  building to make it falsifiable, is open. It matters because Task 5's asynchronous
  `progress()` wording contemplates exactly the world in which it stops being free.
- **O4. Stage 1's poll bound (D1) has no measured basis.** `g` is unmeasured, and so is
  the latency from the service's ready store to a device poll observing it. The bound is
  currently a guess and should be a measurement.
- **O5. `T(n)` enters V1 too, and `PER_ROW_TRANSFER.md` §1.2's two-phase row does not
  model it.** The plan books `T(n)` (OPEN 1, the gather's cost at `count` = 1-6) entirely
  against V2, on the grounds that V2 copies one row per launch. But V1 splits one
  `count = k` launch into `count = h` and `count = k - h`, both smaller than today's.
  The penalty is smaller than V2's and is not zero. **It is not quantified anywhere**, and
  it subtracts from the 35.74 ms ceiling.
- **O6. `c` is bounded to roughly 0.97-1.08 ms/row and not chosen within it.** V1's
  ceiling is `33.9 x c`, so it moves about +-3% on `c` alone. `c` was not measured on
  2026-09-21 (no physical core on divix01 had both SMT siblings under the frozen 10%
  foreign-CPU gate). Whether clause 1's end-to-end arms face the same blocker as the `c`
  microbenchmark is **not established** and should be settled before either is scheduled
  rather than discovered inside a GPU window.
- **O7. Deferral interaction, not traced.** A deferred request is retried under the same
  seq. S2 grants after the defer decision, so a deferred request grants nothing — but
  `deferral_may_retry` (`:2071`) consults `terminal_seen_for(deferred_seq_,
  deferred_gen_)`, and I did not trace what happens when a *previous* generation's hit
  leases are still outstanding on the same ring index. The guard is
  `grant_lanes_locked`'s `if (entry.active) return false` (`:2083`); whether opening the
  entry earlier (S1) widens the window in which that returns false is unsettled.
- **O8. `sim_ack` and `hold_until_ack_lane` do not exist.** The tests above are specified
  to avoid needing them: T1 and T4 read the lease block directly from the test thread, and
  T5/T6/T8/T9/T11 are GPU tests. If a CPU-only simulation of the two-stage chain is
  wanted, `sim_ack` must be built first (`LEASE_PROTOCOL.md` 18.1), and that is not
  costed here.
- **O9. Not checked:** whether any producer other than router-miss via
  `expert_row_plan.py` can put a planned lane outside `protect`. `PER_ROW_TRANSFER.md`
  §3.3 reports `plan_candidates` has no production caller. I did not re-verify that, and
  V1's T3(b) assumes it.
