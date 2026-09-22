# Task 6, V1 (two-phase): implementation checklist

**Companion to** `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, Task 6.
That section's checklist is **not this one** and must not be followed for this work.

Every line number below is against **`f3f66fb2d1`** on `dsv41`. They are given so a
reader can find the code, not so a reader can trust them after a rebase; re-check before
editing.

---

> ## UNPARKED 2026-09-21 — the parking reasoning was wrong
>
> **This task was parked earlier today and is now live again.** The park note that stood
> here is preserved in git history; it is replaced rather than amended because its central
> claim was false and leaving it beside a correction invites someone to average the two.
>
> **1. V1's value was misstated by an order of magnitude.** The note said "V1's honest
> projection was 1.1-1.4% net throughout". That figure is **V2-at-best-order's marginal
> increment over V1**, not V1. V1 two-phase is modelled at **35.74 ms/step, 14.0%**
> (`2026-09-20-storage-cpu-pipeline-v2.md:789`, `PER_ROW_TRANSFER.md:138`). The error
> originated in the lead's brief, not in this file's analysis, and was written down here
> in good faith.
>
> **2. The "saturated link" argument does not bear on V1.** The note parked the task
> because "V1 and V2 are both reordering work on a saturated link". PCIe Gen3 x16 line
> rate bounds **how fast** the gather runs; it says nothing about **when it may start**.
> V1's entire mechanism is starting hit-lane copies before the missed rows have been read
> — 88% of the 35.74 ms is hit-lane hiding. The old note conceded the mechanism in its own
> words ("everything in this checklist moves copy time earlier") and treated it as the
> objection when it is the win.
>
> **3. A trace on today's code shows exactly the serialisation V1 removes.** Node-mode
> nsys capture at `e61c505731`, mirrors on, decode-only window, graph body only:
> `exl3_ram_miss_wait_kernel` **53.5%** and `copy_expert_row_segments_gpu_kernel` **41.8%**
> of in-graph GPU time — **95.3% between them, serialised**, with all compute at ~3%. The
> barrier is stated in the source: `exl3_ram_miss_host.cpp:468-469`, "read() itself
> publishes nothing ... so no row is visible before the whole request is", with every slot
> marked `kReady` together at `:2599`.
>
> **What the old note got right and is retained.** The **Gen3 x16 ceiling is real** and
> the host is an HPE ProLiant DL380 Gen10, so Gen3 is a platform limit rather than a BIOS
> setting: no change makes the gather *faster*, and only fewer bytes per token reduces its
> size. The **`g` lower bound** (`g_a >= 7.26 us`, D1) stands, but it bears on
> **V2-at-best-order only** — it never applied to V1. §2's `release_locked` landmine, §1.3's
> correction and §5's mutant discipline are unaffected.
>
> **Read 35.74 ms as a ceiling, never as an expected gain.** It is
> 33.9 hit lanes x 1.055 ms and "the model assumes every hit copy perfectly hidden"
> (`PER_ROW_TRANSFER.md:236`). It is **modelled, not measured**. §7 O5 further warns that
> V1 pays one extra fixed per-launch cost per reading request that the 35.74 row does not
> model, so the margin is optimistic by an unquantified amount **on top of** being a
> ceiling. §7 O6 puts +-3% on `c` alone.
>
> *Denominators, corrected.* The old note's 391 ms step, its 190 ms / 48.6% NVMe wait and
> its "drive idle about 200 ms of every step" are **pre-mirror single-drive figures** and
> must not be reused. Today's machine runs **~255 ms/step** (3.905-3.933 tok/s mirrored,
> `DSV41_REFERENCE.md` section 19 Task 1 matched baselines). Usefully, V1's model was
> already built against a **254.4 ms** step, so **35.74 ms / 14.0% is calibrated to
> today's step time, not to the stale one.** §6 of this checklist still quotes ~360 ms;
> that is a third measurement and must not be mixed with either.


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

**The single most likely bug in this task, stated here because §4 is what you read while
debugging and this is what you need before you start.** Stage 1 will be written by copying
`exl3_ram_miss_lease_wait_kernel`, and that kernel treats every lane it cannot validate the
same way. In stage 1 the two cases are **opposite**:

- A lane whose `RowResult.ready` is **not set** is **not a failure**. It is a miss lane,
  the service has not published it yet, and it belongs to stage 2.
- A lane that **is** ready but **invalid** — wrong expert, `host_slot` out of range, torn
  seqlock — is a **hard failure**, exactly as today.

In the source you would copy from, both arrive at the same `misses > 0 -> ok = false;
reason = kLeaseReasonIdentity` branch (`:486-491`) and are indistinguishable. Conflating
them fails every mixed request; conflating them the other way turns a genuine identity
violation into a silent second-stage retry. Keep them apart from the first line you write.
D1 restates this where the kernel is specified.

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

3. **The plan's V2 landmine is stated against a throw that does not exist, and the real
   failure is silent.** This is not a footnote and it is not V2-only in the way the plan
   implies, so it has **its own section: §2 below.** Read it before writing any step.

4. **V1 escapes that landmine, but incidentally, and for a narrower reason than the plan
   gives.** Also §2.

5. **V1 *is* exposed to a failure the plan does not name.** Today `grant_lanes_locked`
   runs only when `ok`, so a failed request grants nothing. Under V1 the hit leases are
   granted **before** `read()`, so a read error, a fault or a cancel leaves them
   outstanding. The host already copes — `retire_leases` (`:2126`) resolves acknowledgement
   and terminal mask per lane independently — but the device publishes a **whole-request**
   mask today (`exl3_ram_miss.cuh:513`, `(1u << named) - 1u`). Making that mask partial is
   a device change; `retire_leases` needs no edit for it. One small host edit *is* needed,
   for a different reason: see S4.

6. **`state[kSticky]` is never cleared, and that is by design.** `exl3_ram_miss.cuh` sets
   it at `:217`, `:358` and `:516`; there is no clear site in the kernels, and nothing in
   `ops/moe/exl3_ram_miss.py` re-zeroes the state tensor after `:790`. It is a
   **process-lifetime fail-stop latch**: once this device state has failed once, every
   later request refuses at entry rather than proceeding on a page that may be
   inconsistent. **The missing clear is the feature, not an oversight — do not "fix" it.**
   Adding a clear would silently disarm the fatal path for every request after the first
   failure. Do not use it as `req_failed`; D6 adds a real per-request word instead.

7. **V1 needs no change to `RowReader::read` and no `pack_one` callback.** That hook is
   V2's, for publishing miss rows at their own pack. V1 publishes misses exactly where
   they are published today. If you find yourself editing `read()`, you have drifted into
   V2.

8. **`sim_ack` and `hold_until_ack_lane` do not exist** anywhere in `python/` or `test/`.
   Both are `[P]` proposals in `PER_ROW_TRANSFER.md`. The tests below are specified so
   that the host-side ones need neither.

---

## 2. The `release_locked` landmine: unreachable today, incidentally, and how it arms

> **This hazard has a home outside this parked file.** It is a property of the lease path
> as shipped, so it is recorded as a **standing risk** in
> `2026-09-20-storage-cpu-pipeline-v2.md`, under its own heading before Task 0 rather than
> under Task 6, with the three independent routes that reached it. **That entry is not
> parked and does not depend on Task 6 resuming.** What follows here is the full
> derivation it points back to.

The plan records this as an "implementation landmine, V2 only" and states it in terms of
a throw. Both halves need correcting, and the correction makes it more dangerous rather
than less. It is given its own section because a hazard whose only defence is that nobody
has yet written the line that triggers it does not survive being buried in a list.

**What the plan says.** "Publishing miss rows early lets a row be `kReady` and leased when
a *later* row of the same request fails. `serve()`'s `!ok && !cancelled` cleanup calls
`release_locked` on every slot, and Task 5 makes releasing a leased slot throw, so the
cleanup must skip published lanes or the service thread throws on any mid-request failure.
V1 does not hit this: its hit lanes are already `kReady`."

**What the tree says.**

1. **The throw is in the wrong function.** `release(row, slot)` (`:1975`) does check, and
   raises "release of pinned slot N while it is leased" at `:1982`. But `release()` is the
   *public* pin/unpin API that Python calls. `serve()`'s cleanup calls **`release_locked`**
   (`:2439`), which is ten lines long and checks nothing: it publishes map `-1`, clears
   `slot_to_expert` and sets `state = kFree`. A leased slot goes through it silently.
2. **Nothing downstream catches it either.** `take_slot_locked`'s first loop (`:2408-2409`)
   returns the first `kFree` slot it finds and **never consults `leased_locked`** — the
   lease check at `:2415` guards only the `kReady` eviction scan below it. So the freed
   slot is handed to the very next request that needs one.
3. **The failure is therefore two silent corruptions, not one loud stop.** The GPU may
   still be reading that slot under its lease while the next request's bytes are written
   into it — **wrong bytes, no diagnostic**. And when the device finally acknowledges or
   the terminal voids that lane, `release_lease_locked` (`:2169`) decrements
   `tier.leases[]` on a slot that now belongs to a *different* expert, corrupting that
   slot's lease count. The only counter that would ever notice is
   `kLeaseDoubleSignal`, and only by accident.

**Why V1 does not reach it today.** Not because "its hit lanes are already `kReady`" — that
is true but is not what saves it. The operative fact is checkable: `serve()`'s two
`release_locked` call sites (`:2541` in the reservation bail, `:2604` in the post-read
publish loop) **both iterate `slots`**, and `slots` only ever receives entries from the
`take_slot_locked` loop at `:2526-2537`, i.e. **newly taken slots for experts in
`missing`**. A hit lane's expert is by definition not in `missing`, so a hit lane's slot is
never in `slots`, so no `release_locked` in `serve()` can reach it. A second fact backs it
up: `:2490` pushes every `lane_expert` into `wanted`, and `take_slot_locked` is called with
`fallback = false`, so a `wanted` expert is never its own request's victim.

**This is incidental safety, and incidental safety is what gets deleted by someone who did
not know it was load-bearing.** Today's wider escape — that `grant_lanes_locked` runs only
when `ok`, so no lease exists at all when the cleanup runs — **is exactly what V1 removes**
(§1.5). After V1, leases *do* exist while the cleanup runs; the only thing standing between
them and `release_locked` is the `slots`/`missing` disjointness above.

**The rule this imposes on any future change.** Stated as a rule because the condition is
general and the next person to hit it will not be implementing V1:

> Any change that moves a lease grant before `read()` — V1, V2, or a promotion holder —
> must either **add a lease check to `release_locked` (`:2439`) and to
> `take_slot_locked`'s free-slot loop (`:2408-2409`)**, or **prove that neither can reach a
> leased slot.** V1 takes the second route, and §3 S6 and §5 T3/T4/T4b are that proof.
> V2, which publishes miss rows early, cannot take the second route: its miss rows are in
> `slots` by construction, and it must take the first.

**Corroborated from an independent direction.** While this section was being written, the
agent closing **Task 5 item 5** found by mutation that the service acting on a `Terminal`
**without checking its generation** survives the entire existing suite, and that a stale
terminal from an earlier lap would release a lease a GPU may still be reading. That is the
same unguarded invariant as this section's — *a lease released while its reader is live,
caught by no existing test* — reached from the opposite end of the protocol. See Task 5
item 5's **R3 mutant**. Whoever implements V1 should read both: this section covers the
release path, R3 covers the signal that triggers it.

**The exact edit that would arm it**, so a reviewer can recognise it: pushing a hit lane's
slot into `slots` during the reservation loop (a plausible refactor, since `slots` looks
like "the slots this request touched"), or widening the demand path's cancel predicate so
`cancelled` becomes reachable for demands and the `:2604` branch runs with hit leases held.

---

## 3. Service-side checklist (host C++, `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`)

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
      Do **not** release them on the host: that is exactly the silent-corruption path of §2.

- [ ] **S6. Make §2's two facts assertions rather than arguments.** Add a debug
      assertion at `:2541` and `:2604` that the slot being released is unleased. Do **not**
      add a lease check to `release_locked` itself and do **not** route the cleanup through
      `release` — that is the first route of §2's rule, which V2 must take and
      V1 need not (open question O2).

- [ ] **S7. Separate the counters.** `counters_[kLeasesGranted]` is incremented once per
      request at `:2119`. Split it, or add a `hit_leases_granted` counter, so that an arm
      or a test can show the hit group was granted **before** `read()` at all. Without a
      counter that distinguishes the groups, a build that publishes nothing early passes
      every timing test below by falling through to the batched path.

---

## 4. Device-side checklist (CUDA, `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`)

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
      - *Contiguity, and the ruling behind it.* The copy kernel takes base pointers plus a
        count, so stage 1's lanes must be contiguous. **Compaction inside the stage kernel
        is the design; `ord`/`h` written by the post kernel is the recorded alternative,
        rejected for now.** Ruled 2026-09-21. Two notes, because the documents disagree
        about whose mechanism `ord` is:
        - `ord` **is V1's**, not V2's. `PER_ROW_TRANSFER.md` §3.2 says so in those words
          ("Consequence for **V1's** order array `ord` (section 4.2)"), and §4.1's table
          gives V1 the ranges `[0, h)` / `[h, k)`, which *is* `ord` ordering. The plan's
          Task 6 warning that "there is no `ord` array in [the checklist and pseudo-code]"
          is a criticism of the **V2** checklist, not a prohibition on V1 using one. An
          earlier framing of this checklist's brief had it the other way; it was wrong.
        - **Why rejected for now:** `ord` adds a field to the lease protocol, which is the
          part of this system the most effort has gone into proving properties about, and
          it buys only poll-read savings. Compaction keeps the hint out of the protocol
          entirely. §4.2's rationale for `ord` is untouched and still correct: it is a
          pure performance hint, benign when wrong, because correctness lives in the
          per-lane `RowResult`.
        - **The condition that reopens it**, written so it survives both a new `g` figure
          and a change to the wait design: **`ord` deserves a second look when the poll
          cost of the wait design actually being built is shown to be a material share of
          the per-stage cost `g`.** Not "when `g` is high" — `ord` saves *poll reads*, so
          what reopens it is poll reads mattering, which depends on the wait as much as on
          `g`. That is a trigger, not a dead end, and it does not expire when someone
          redesigns the wait.
        - **Record every `g` figure with its direction.** As of 2026-09-21 the only reading
          is **`g_a >= 7.26 us`, a LOWER BOUND, not an estimate** — the stand-in kernels are
          strictly simpler than the real per-stage ones (the stand-in `W_s` polls four words
          and stores; the real one also decodes fatal, shutdown, `RowResult.ready` and the
          seqlock, and computes per-lane `go`), and ready-at-launch is the best case for
          polling. The bound sits **above** the 6.98 us crossing and 0.7 us below the low
          end of the assumed 8-14 us range, so **it supports neither side of the `ord`
          question and must not be read as "`g` came in low".** It also holds only for the
          design **as specified at `p = 4`**: the same harness measures 4.90 us for a
          one-word `W_s`, so a cheaper wait is not bounded by this number at all.
          A bound quoted without its direction gets read as an estimate — that has already
          happened once to this figure, which is why the direction is written next to it.
        - The price of compaction is a real failure mode — an indexing error that sends a
          lane's bytes to the wrong destination slot — which `ord` would not have. Test T9
          exists to pay it.
        - **Two concrete instances, both hit while implementing this on 2026-09-21.** They
          are recorded here because T9 tests for the second one while the instructions
          warned of neither, so both were found by the implementer rather than by the file.
          1. **Compaction breaks the ack-to-lane mapping.** The ack kernel keys its `LaneAck`
             word by `threadIdx.x`, which after compaction is the *compacted* position, while
             `retire_leases` reads acks *by lane*. Left alone this acknowledges the wrong
             lease. It needs an explicit per-stage `origin[]` array.
          2. **Destinations must be compacted with sources.** `copy_expert_row_segments_gpu`
             takes `source_rows` **and** `destination_slots` index-aligned; compacting only
             the sources sends a lane's bytes to another lane's slot — which is exactly
             T9's mutant, arrived at by accident instead of on purpose.

- [ ] **D2. Change `exl3_ram_miss_lease_wait_kernel` (`:382`) into stage 2.** It cannot be
      reused unmodified.
      - **This instruction conflicts with §6 and §6 wins.** That kernel *is* the M1 arm.
        Mutating it in place deletes M1 from the build, and §6 requires M1 and M2 measured
        in one build. **Resolution taken 2026-09-21:** stage 2 was added *alongside* the
        existing kernel, behind `SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE` (default off), so
        M1 is unchanged and both arms exist in one binary. Same reason for a separate
        stage-ack kernel. Read the bullets below as describing stage 2's required
        behaviour, not as licence to edit the M1 kernel.
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
      and **one** `lane_ctx`; it needs one per stage. **Correction (verified 2026-09-21):
      there is no `_owned_tensors`** — the name appears nowhere in `python/`, `test/` or
      `scripts/`. The thing at `srt/layers/moe/exl3_ram_miss.py:712` is a *local* `owned`
      list built inside `Exl3RamMissService._quarantine`, and the stated rationale was also
      wrong: these are attributes on a live object, so nothing frees them out from under the
      captured graph. The real reason to add the new per-stage buffers to that list is
      unclean-shutdown quarantine.

- [ ] **D8. Register the new kernel names** in the `names` tuple at
      `python/sglang/kernels/ops/moe/exl3_ram_miss.py:740`. Omitting one fails at JIT load
      with an error that will not say why.

---

## 5. What must be tested, and the mutant that proves each test is not vacuous

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
| **T3** | A hit lane's slot is never its own request's victim. | Tier at capacity exactly `k`, every slot resident, the hit expert among them, the reservation needing a victim. | **Two mutants, both required.** (a) Force `listed(protect, expert)` false in `take_slot_locked` — per the plan's SAFETY passage this may stay green through recency, and that is the point of running it. (b) Delete `:2490`, the `lane_experts -> wanted` loop; a lane expert that is not routed then becomes a legal victim of the request leasing it. (b) is the one matched to the claim. **Resolved 2026-09-22 (T3 written):** (a) stayed green over 3 runs, exactly as predicted -- the reservation loop bumps the hit slot's stamp before eviction, so recency alone spares it with protection fully disabled. (b) killed, via `waited.status == 2`: the unprotected hit slot became the LRU victim, so the hit-lease grant failed and the request was answered failed. |
| **T4** | A failed request voids the hit leases and **releases no slot**. | `inject(fail_reads = True)` on a mixed request whose hit lane was granted at S2. Assert: right after the request answers, `slot_info` shows the hit slot still `kReady` with `leases == 1`; the device terminal names the hit lane; after it, `leases_voided == 1` and `leases == 0`. | Add `release_locked` for the hit slot to the `!ok` path. Must go red on `leases`; a **second** assertion must catch the consequence — issue a request for a different expert and assert it does not land on that slot. One assertion alone would pass on a build that corrupts the count without reusing the slot. |
| **T4b** | **A hit lane's slot is never in `slots`.** This is the whole proof that V1 escapes §2's landmine, and it is the one fact in this checklist that is load-bearing for *correctness of bytes* rather than for performance. | Mixed request, at least one hit and one miss. Instrument or assert that every element of `slots` corresponds to an expert in `missing`, and that the hit lane's slot is not among them. Run it on the reservation bail path too (`inject` a reservation failure) so `:2541` is exercised with a hit lease held. | Push the hit lane's slot into `slots` in the reservation loop — the plausible refactor §2 names. The test must go red on the membership assertion **and** T4's slot-reuse assertion must go red as well. If only the membership assertion fires, the consequence is untested and the pair should be treated as one vacuous test, not two. |
| **T4c** | An ungranted lane keeps the ring entry open. | Between S2 and S3 a miss lane is granted-pending. Drive `retire_leases` while a request is in that state (the service thread is inside `read()`; call it from the test thread) and assert `entry.active` is still true and the ring index is not reused. | Revert S4 — restore the `open` loop at `:2156-2157` to count only `state == 1`. Must go red on `entry.active`. Without this the S4 edit is an unjustified change, and an unjustified change is one a later cleanup deletes. |
| **T5** | The partial terminal mask names **only** unacknowledged lanes. | GPU. Stage 1 copies and acknowledges; stage 2 fails on an injected read failure. Assert `skipped_mask` has the hit lane's bit **clear** and the miss lane's set, and `kLeaseDoubleSignal == 0`. | Restore the whole-request mask `(1u << named) - 1u` at `:513`. **Resolved 2026-09-22 (T5 written):** the mutant goes red on the mask bit, confirmed. The second detector does **not** fire, and the claim that it would is **retired 2026-09-22**, with the mechanism established rather than assumed. Measured `mask = 3`, `lease_double_signal = 0`. The first explanation offered here -- the global `lanes_outstanding_ == 0` early-out at `exl3_ram_miss_host.cpp:2212` -- is real but **not** the operative cause: a retry with a second, deliberately unacknowledged lease held open on the *same* device kept `lanes_outstanding_` provably above zero (`granted - acked - voided == 1` at the moment the terminal landed) and the detector still stayed 0. The operative gate is **per-entry**. `retire_leases` skips any entry with `!entry.active` (`:2216`), and it clears `active` at the end of a pass once no lane is left in `state == 1` (`:2244-2246`). The double-signal check itself (`:2236`) needs the ack and the wrong terminal visible within **one** poll -- `state` flips 1 -> 2 at `:2231` and the check reads `state == 2 && voided` in the same iteration, so a single pass seeing both *would* count it. It never does, because **stage 2's entire wait sits between `stage_ack(1)` and `finalize`**: the ack is visible for the whole of that wait, so a poll lands in the gap, releases the lane, closes the entry, and the terminal is compared against nothing. That is structural to the two-phase pipeline, not a timing fluke, and no lease held elsewhere can change it because `active` is per-ring-entry. The mask assertion alone carries the test. Separately, the `go_1` 1 -> 0 perturbation recorded earlier **did not reproduce** on a single device and is confirmed an artifact of two devices contending over one lease block, which production never does. **T5 was also flaky, found and fixed 2026-09-22 (`19eb5837c5`):** measured **1 failure in 12 runs** at `2e4f2f0e3c`, with none of the progress-loop work present, showing `go_1 == 0` and `leases_voided == 1`. Nothing made stage 1 reach its poll before the host finished the request the test deliberately fails. The cause is W1's early exit on the request being served, whose comment argues that once served every lane is published, so a later poll can discover nothing new -- a premise that does **not** hold for a FAILED request, whose hit lanes are voided rather than published. The bounded poll is not what binds: raising `poll_bound` from 256 to 2000000 still failed 1 run in 12. Fixed by injecting a read delay alongside `fail_reads`, which runs before the `fail_reads` check and so holds the request open after S2 has published the hit lane, mirroring the injected delay T9 needed for the same class of race in the opposite direction. This reaches past this branch: T5 is merged Task 6 evidence, so for as long as it flaked its recorded mutant kills could not be fully relied on. |
| **T6** | `keep` has exactly one writer. | GPU. Force a VIOLATED acknowledgement in stage 1 (bump the slot generation between grant and acknowledgement) and let stage 2 succeed. Assert `keep == 0` and no fused output. | Restore `keep[0] = 1.0f` in the stage-2 wait. **Resolved 2026-09-22 (T6 written):** killed as predicted, at the intermediate check before `finalize()` runs. Note the mutant had to *add* a `keep` pointer to `rest_wait` (wrapper and ops.py included), because the fixed code has none to reuse -- which is itself the evidence that D2 landed. |
| **T7** | One request deadline, not one per stage. | Delay both stages to 0.9x the timeout; the request must fail at ~1x, not ~1.8x. | Give each stage its own `start + timeout_ns`. **Resolved 2026-09-22 (T7 written, then diagnosed):** killed at 14.23 s against a 0.56 s bound. That magnitude was initially unexplained and a read-drain hypothesis was recorded here; **the hypothesis was wrong and is withdrawn.** Measured by varying only `delay_s`, with `timeout_ms = 400` and `poll_bound = 100_000_000` fixed:

| | `delay_s=1.0` | `delay_s=5.0` |
|---|---|---|
| baseline | 0.4250 s | 0.4265 s |
| mutant | 15.35 s | 0.8262 s |

Read-drain predicts elapsed tracking `delay_s`; it does not, and is if anything inverted. The baseline is delay-independent at ~1x the timeout, which is the shared deadline giving up regardless of read speed. The mutant at `delay_s = 5.0` gives 0.8262 s, almost exactly the 0.4 + 0.4 the two-independent-deadlines model predicts. **The row's original "~1x vs ~1.8x" framing is correct and the test measures what it claims.**

A separate intermittent ~14-15 s band appears on roughly half of *mutant* trials at both delay values (14.23 s at `delay_s = 5.0`, 15.35 s at `delay_s = 1.0`) and has no confirmed mechanism. It is **mutant-only and cannot affect production**: the baseline never shows it, and the shipped path stores one deadline in the post kernel (`exl3_ram_miss.cuh:324`) and *loads* it in `rest_wait` (`:634`), while the mutant substitutes a fresh in-kernel `global_ns()` read. A `%globaltimer` cold-read or clock-domain skew on that fresh read is the leading unverified candidate. Recorded rather than chased, because the code it afflicts is throwaway. |
| **T8** | The captured graph is the linear chain D7 builds. | Enumerate kernel-node dependencies with `cudaGraphGetEdges`; assert post → W1 → C1 → A1 → W2 → C2 → A2 → F → fused. | Capture A1 on a second stream. (`PER_ROW_TRANSFER.md` §5.2 item 1 requires this be checked, not assumed.) **Resolved 2026-09-22 (T8 written):** killed -- the fork/join inserts event nodes and the graph stops being a simple chain. The captured graph is **9 nodes, not 8**: `post`'s `_stage_planned` is a device-to-device MEMCPY node, not a kernel, and leads the chain on every replay. The `-> fused` clause is a **separate** test, since `post()`'s own capture ends at F: it captures the real `_apply_graph` and asserts no node is incomparable with F. Its mutant is a **diamond** -- fork before `gather`, run `fused.run` on the side stream, join after -- which captures cleanly (a join satisfies capture; an *unjoined* fork is refused by CUDA with `cudaErrorStreamCaptureUnjoined` and so kills every such test uniformly, via a CUDA error rather than any assertion). It killed on the incomparability assertion: 26 fused-consumer kernels became siblings of F. That is the real risk -- a future optimization overlapping fused compute with the chain's tail. |
| **T9** | Output parity. | Destination rows byte-equal and fused output bitwise equal against **M1** (the Task 5 lease-mode batched arm) on fixed routes and seeds, eager and graph. | Swap two lanes' destination slots in stage 1's compaction. This is the failure mode compaction introduces and `ord` would not — it is the price of D1's compaction ruling and must be paid in a test. **Resolved 2026-09-22 (T9 written):** killed on the byte-equality assertion against a fresh checkpoint read, before the M1/M2 comparison runs. No tolerance anywhere in the file; equality is `torch.equal` and `.view(torch.uint8)`. Parity alone would have survived stage 1 becoming a no-op, so the test also asserts stage 1 liveness and that a never-resident expert is not claimed. The latter needed an injected read delay to be **true**: stage 1 does not distinguish "resident before this request" from "published while I was still polling" (that is D1's design), and against a small fake checkpoint a miss's publish routinely lands inside the poll window. Re-running the mutant with both new assertions present still fails on byte equality, not on them. |
| **T10** | An all-miss request does not pay the read wait twice. | Every lane misses. Assert stage 1 commits `go_1 == 0` promptly, with a wall-clock upper bound. | Remove D1's poll bound and let stage 1 spin to the deadline. **Resolved 2026-09-22 (T10 written):** killed as predicted -- elapsed 2.0002 s (the full deadline) against a 0.3 s bound, at the real default `poll_bound = 64`. |
| **T11** | The batched path survives as stage 1 covering every lane. | All-hit request: assert `go_1 == k`, `go_2 == 0`, the empty stage acknowledges nothing, and a poisoned source slab is not read by stage 2. | Make an empty stage's acknowledgement kernel acknowledge lane 0. **Resolved 2026-09-22 (T11 written):** kills, but **not** on `leases_acked`, and the row's expectation was wrong about which signal fires. `leases_acked` bumps by exactly the expected +3: the bogus lane-0 write's ring index, computed from a zero/underflowed generation, lands on a ring slot `retire_leases` does not track as active, so it is silently ignored for counting, and `lease_double_signal` stays 0 too. What goes red is `keep == 1.0` -> `0.0`: the same write's slot-generation mismatch sets the device-side `violated[0]`, and `finalize()` correctly refuses to serve a violated request. Accepted as a stronger signal than the named one -- it shows the consequence, not just a counter. Also note the mutant corrupts any *setup* pass that uses the two-phase chain, so T11 stages residency through the batched chain, which shares no kernel with `stage_ack`. |

Existing files to extend rather than duplicate:
`test/registered/unit/kernels/test_exl3_ram_miss_leases.py`,
`test_exl3_ram_miss_lease_service.py`, `test_exl3_ram_miss_lease_thread.py` (T1-T4, T7 host
half); `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`,
`test_exl3_ram_miss_graph_gpu.py` (T5, T6, T8, T9, T11).

---

## 6. Reporting requirement and baseline

**Naming, fixed 2026-09-21 after it caused a real implementation deviation.** Earlier drafts
of this file called the measurement arms A0/A1/A2 while the pipeline chain at §3 calls its
kernel stages `post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F`. **`A1` and `A2` therefore meant
two unrelated things in one document** — an acknowledgement kernel and a measurement arm —
and the falsification criterion below is stated as a ratio against one of them. The arms are
now **M0/M1/M2**; the stage names are unchanged, because they are in the implemented code.
Read any surviving `A1`/`A2` in this file as a *stage*.

**Baseline is M1 = Task 5 lease-mode batched**, not today's unleased path. Lease mode arms
every `count > 0` record, so all 40 layers pay a service round trip that M0 does not.

**Report all three differences, each with an interval: M1 v M0, M2 v M1, M2 v M0**
(M0 = today's unleased path, M1 = Task 5 lease-mode batched, M2 = this mechanism). M2 v M0
alone credits V1 with the cost it inherits from Task 5; M2 v M1 alone hides a possible loss
inside Task 5. Neither is "the Task 6 result" on its own.

**Net out OPEN 11: 0.68 ms/step at 40 layers, 17 us per armed layer**, re-measured
2026-09-21 on an exclusively held card (`analysis/dsv41-drive/open11/results.md`). That is
**~61% of the 1.114 ms `G*`** Task 6 is trying to win. `LEASE_PROTOCOL.md` §17.2 now
carries this figure and flags itself as the paragraph Task 6's baseline requirement is
stated in (`c3b2254dd1`); the first draft of this checklist reported it as stale, and it
was, until that commit. Use §17.2's wording, not an older copy.

**The two measurements are not reconciled, and that matters here.** The first,
hand-written-step measurement gave ~8 us / ~0.32 ms; the re-take through
`Exl3RamMissRowBackend` in a CUDA graph gave ~17 us / ~0.68 ms, and `LEASE_PROTOCOL.md`
§15 says in terms that the two "are not reconciled" (its §20.2m). So the quantity this
task must net out is known to about a factor of two, by a discrepancy nobody has explained.
Report the netting with the figure used named explicitly, and do not average them.

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
is falsified if the untraced tok/s ratio V1 / M1 (M1 = the Task 5 lease-mode batched arm,
not the stage-1 acknowledgement kernel) has a 95% interval containing 1.0, or if
the point estimate is below 3%. The modelled prediction is 20.8 to 35.7 ms/step, 8% to 14%
of the step — **modelled, not measured**, a ceiling by construction (35.74 = 33.9 hit lanes
x `c` = every hit copy perfectly hidden), and nothing in it models a hit copy that fails to
hide.

**The Gate's clause 3 still applies.** V1 reads no `slot_map` before `demand_done` —
correctness comes from the per-lane `RowResult`, which is why it is exempt in fact. State
that exemption explicitly in the acceptance record rather than leaving it inferred, since
the clause is a refusal.

---

## 7. Open questions — stated as open, not guessed

*(O1, compaction versus `ord`, was open in the first draft. It was ruled on 2026-09-21
in favour of compaction and now lives at D1, with the condition that would reopen it.)*

- **O2. Should `release_locked` grow a lease check anyway?** §2's rule lets V1 take the
  proof route, so it does not need one. But a check in `release_locked` (`:2439`) and in
  `take_slot_locked`'s free-slot loop (`:2408-2409`) would convert §2's hazard from silent
  to loud **for every future caller at once**, including V2, which cannot take the proof
  route. It is cheap and it is not V1's to decide. Recommended to Task 5's owner; not
  taken here.
- **O3. The same-`mutex_`-hold claim (T2) has no falsifying mutant** with one service
  thread. Whether that is acceptable, or whether a test-only second thread is worth
  building to make it falsifiable, is open. It matters because Task 5's asynchronous
  `progress()` wording contemplates exactly the world in which it stops being free.
- **O4. Stage 1's poll bound (D1) has no measured basis.** `g` is **not measured** — the
  only reading is the lower bound `g_a >= 7.26 us` recorded at D1, which is not an
  estimate and settles nothing — and neither is the latency from the service's ready store
  to a device poll observing it. The bound in D1 is currently a guess and should be a
  measurement. `c` likewise remains **NOT MEASURED under its pre-registration**, and that
  run is parked, so **no per-launch or per-row cost figure in this checklist is gated.**
- **O5. `T(n)` enters V1 too, and §1.2's two-phase row does not model it.** Raised while
  drafting this checklist and **recorded centrally in `PER_ROW_TRANSFER.md` at
  `8f92f922af`**, which is the statement to cite; it is not repeated here. In short: V1
  pays **one** extra fixed per-launch cost per reading request where V2 pays about
  `k - 1`, and the 35.74 ms row models a single `c` per row with no per-launch term.
  **No number is put on it, deliberately** — the magnitude is what the `c` measurement is
  being taken to find. Two consequences for this checklist: V1's margin is optimistic by
  an unquantified amount on top of already being a ceiling, and **clause 4's `g`
  measurement will not catch it**, because `T(n)` is a copy-kernel cost and `g` is a stage
  cost.
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
