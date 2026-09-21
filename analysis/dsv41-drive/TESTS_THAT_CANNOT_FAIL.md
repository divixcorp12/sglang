# A sweep for tests that cannot fail: the dsv41 exl3, ram-miss and expert suites (2026-09-21)

Five times in one day this team found a check that could not fail: `pack_one`'s
`filled >= needed` coverage check, the generation gate that printed `GENERATION unknown` then
`VALID`, the `PROMOTION_ASYNC` M0 test, `test_a_failed_read_publishes_none_of_the_rows_it_had_already_packed`,
and a `kLease*` agreement test that matches nothing. This sweep looks for the rest.

Scope: `test/registered/unit/kernels/` and `test/registered/unit/layers/moe/` for the exl3, ram-miss,
expert, engram, residency, mirror and uring files, plus the top-level tests and manual dsv41 unit tests
that name these components (82 files). Read-only: nothing was fixed, and `exl3_ram_miss_host.cpp` was
not touched. Working-tree line numbers; the host `.cpp`, `ops/moe/exl3_ram_miss.py`, `environ.py` and
two moe files were dirty while this was written.

## Result

About 1,140 tests were read across five slices. **21 were proven unable to fail for the reason they
claim** (about 2%), and 9 groups of weaker candidates are listed as needing one mutant to settle. The
rest read as sound. That is a real and useful result, so say it first:

- **The byte-exactness suites are not vacuous.** The reader, mirror-source, read-split, copy-kernel,
  doorbell-copy, host-tier, hot-cache-publication and fused-route-plan tests use distinct random rows,
  non-identity permutations, zeroed or sentinel destinations, independent reference implementations, and
  poison bytes that land in bytes the code reads. Where a fault is injected, in nearly every case it
  fires after the state change the test then asserts.
- **The problems cluster in three places:** tests of failure paths (a fault that never reaches the code
  under test), tests of guards that are cheap to satisfy some other way (an identity, a redundant second
  guard, an expert-id check standing in for a generation check), and tests of an instrument's
  self-report. Those are the ones the top four findings below are.

**Method, and its limit.** Five reviewers each read a slice against the production code and were told to
report only what they could prove from source (quote the short-circuit, the empty collection or the
hardcoded value, and name the production mutation the test would miss), listing everything else as
"weak but unproven". I then re-read the top findings against the source myself. **Each finding below is
marked `[re-verified]` (I read the test and production lines and they hold) or `[reviewer-established]`
(the reviewer's proof, not re-read by me).** No mutant was run anywhere: every "would stay green" is a
source argument. The mutants worth running are listed at the end.

## Tier 1: byte-safety and certification

### 1. `test_a_failed_read_publishes_none_of_the_rows_it_had_already_packed`, `test_exl3_ram_miss_thread.py:436` `[re-verified]`

The guarded invariant: a demand whose read fails after some rows packed publishes none of them.

The test injects `host.inject(fail_reads=True)` and asserts nothing published. In
`exl3_ram_miss_host.cpp:2020-2023` that fault sets `ok = false; status = kStatusFailed` *instead of*
calling `reader_.read(...)`, so `read()` never runs, no row ever packs, `packed` is empty (it is cleared
at :2013 and refilled only inside `read()`), and "nothing published" is true by construction.

**The mutant that escapes the whole slice**: in `exl3_ram_miss_host.cpp:2054`, widen
`if (ok || (cancelled && i < packed.size() && packed[i] != 0))` to
`if (ok || (i < packed.size() && packed[i] != 0))`, so rows packed before a hard failure are published.
Every test in `test/registered/unit/kernels/` stays green. Nothing else kills it: the tier test at
`test_exl3_ram_miss_tier.py:128` fails on row 0 of one row (a row whose extent fails is never packed),
`:120` and `thread.py:417` use the same `fail_reads` or never reach `read()`, the split tests never pass a
`packed` pointer, and the cancelled-advisory test at `thread.py:361` has `cancelled` true, so `packed[i]`
decides there and a hard failure is not covered.

**This converges with a finding reached independently earlier today.** Two agents searched for consumers
of the "a failed demand publishes no row" guarantee and concluded, by a *source* argument about this one
test, that it is unpinned in both directions. This slice arrives at the same hole from the *mutation* end
and supplies the line and the edit. Two methods meeting at the same place is stronger than either.

**Why it matters more than one test.** That mutation is almost exactly the change Task 6's V2 proposes to
make on purpose (section 7: rows that packed whole stay published on a failed demand). So the behaviour
change is currently unfalsifiable in both directions: neither the guarantee it would remove nor the
guarantee it would add is pinned. It also supports the existing conclusion that a mid-read fault injector
at the tier is a prerequisite: `RamTier::inject` exposes `fail_reads`, delay and abandon, and none produces
"some rows packed, then a failure".

Test-side repair the reviewer suggested (untried): truncate the source file to end inside expert 2's
superset so rows 0 and 1 pack and row 2 fails, then assert on rows 0 and 1. A mid-read injector is the
cleaner fix.

### 2. The stale-ticket generation gate, `test_expert_hot_cache.py:311` `[re-verified]`

`test_reservation_hides_victim_until_matching_generation_is_ready`. The guarded invariant: a stale ticket
cannot publish or recycle a slot that has since been re-reserved. `_ticket_matches`
(`expert_hot_cache.py:287-295`) checks both `_slot_generations[slot] == ticket.generation` (:292) and
`slot_to_expert[slot] == ticket.expert_id` (:293).

The test's "stale" ticket is `first` (slot 0, expert 2) and the replacement is (slot 0, expert 3), so the
expert-id comparison already returns False. `assertNotEqual(first.generation, replacement.generation)`
(:327) shows only that the generation bumped. No test in the slice re-reserves the **same** expert
(the ABA case), and none calls `retire`, `cancel` or `begin_loading` with a stale ticket at all.

**Mutant missed:** delete the generation comparison at `expert_hot_cache.py:292`. Every test stays green.
This is a stale-ticket guard nobody has seen fire, the same class as `pack_one`'s coverage check before
yesterday. It is also the guard `PROMOTION_ASYNC` leans on ("generation-qualified slot tickets").

**The fixture change that makes it discriminating:** reserve expert 2 into slot 0 (`first`), `cancel` it,
reserve **expert 2 into slot 0 again**, then assert `publish_ready(first)` (and `begin_loading`, `cancel`,
`retire` with `first`) return False while the new ticket's calls return True. Only when the expert is the
same does the generation comparison decide, so that is the only case where it does any work.

### 3. The mirror verifier's "O_DIRECT" is bound to its input, not to its reads: `test_exl3_verify_expert_mirror.py:557`, `:571` `[re-verified]`

The verifier is the instrument that certifies mirror bytes match. Its verdict line says "read with
O_DIRECT (page cache bypassed; the bytes came from the drives)". That text comes from `result.direct`
(`verify_expert_mirror.py:668`), which is set from the flag (`:483`), not from how the rows were read.

The test asserts only `main(...) == 0` and `"O_DIRECT" in out` (:557-568), and skips on any filesystem
that cannot O_DIRECT (tmpfs). `:571` checks only that argparse sets `buffered`. Every other verifier
fixture passes `direct=False` (`:58`).

**Mutant missed:** change `direct=direct` to `direct=False` at `verify_expert_mirror.py:519` (mirror
sources) and `:534` (source rows), leaving `:483`. Every test stays green, and a buffered run prints
`reads: O_DIRECT` and exits 0. A buffered run reads page-cache contents, not what the drive holds: a
mirror whose on-disk bytes are wrong but whose pages are cached from the write that created it would verify
clean. This session established with `dd iflag=direct` and a buffered control that O_DIRECT genuinely
bypasses the page cache on both xfs and ext4 here, so the distinction is real.

**Production does thread `direct` correctly**, and the recorded full run was direct. Read in the source:
`main` -> `verify_mirror(direct=not args.buffered)` (`:836`; the default is `direct=True`, `:405`) ->
`Exl3MirrorRowSource.for_mirrored_layer(direct=...)` (`:519`) and `Exl3ShardRowSource.for_layer(direct=...)`
(`:534`) -> `shared_row_reader(layout, direct, source_root)` (its cache key includes `bool(direct)`, so a
buffered reader cannot be silently reused) -> `Exl3RowReader(direct=...)` ->
`UringFileReader.open(path, direct=self._direct)` -> `open(... O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0))`
(`uring_file_reader.cpp:115`). The design already treats buffered as never a verdict (exit 3, labelled
"BUFFERED (not a drive verification)").

**The one full run we rely on.** Launched 2026-09-19 22:03 CDT from wt-dsv41 with
`--source /mnt/nvme2/... --roots /mnt/nvme0/dsv41_flash /mnt/nvme4/dsv41_flash --keep-going`: no
`--buffered`, no `--layers`, no `--sample-experts`. Log at `divix01:/tmp/cc-verify-mirror.log`:
`MODE: FULL ... 15360 of 15360 expert rows (204.527 GB) per root`, both roots PASS with 0 mismatching,
`VERDICT: VERIFIED`, elapsed 438.4 s. `verify_expert_mirror.py` and every file on the read path are
byte-identical at the commit that ran (`ad5d795cf`) and today.

> **What we do NOT have.** The header text alone would not prove the run was direct, because that text is
> exactly what the test cannot bind to behaviour. What supports it is source: the two `direct=direct` sites
> and the C++ open flag are the same lines at `ad5d795cf` as today. It is corroborated, not measured, by
> throughput: per-root read time 91.0 s and 88.4 s for 204.527 GB is about 2.25 and 2.31 GB/s, which fits
> drives and not memory, and 204.5 GB per root cannot sit in a 188 GB machine's page cache. There was **no
> independent measurement during that run**: no `/proc/diskstats` delta and no `fincore` residency before
> and after, the way the `dd` control worked. So the strength is "proven by source at that commit,
> corroborated by throughput, not measured", and it should be cited at that strength. The parity claim
> that the mirrors, the mirror reads and Task 0's byte parity rest on has not been independently measured.

**Ranking note.** This is a bad test over correct behaviour, but on the instrument that certifies bytes,
and its consequence is that the certification's self-report is unbound. Do not let it sink as trivia.

**The tests that would make it real, in order of importance:**
1. **A spy on `UringFileReader.open`** (or a wrapper of `shared_row_reader`) while running
   `verify_mirror(direct=True)`: assert the spy was called for the source shards and for every mirror
   root's files (non-empty, and the paths cover each root), and that every call had `direct is True`; run
   again with `direct=False` and assert all False. This runs everywhere including tmpfs, kills the mutant,
   and **pins `:519` and `:534` separately**: mutating only `:519` fails on the mirror paths, only `:534`
   on the source paths.
2. **A real-filesystem control** (skip only if an O_DIRECT open fails): write fixture files,
   `posix_fadvise(DONTNEED)`, assert `fincore` residency 0; run direct=True and assert residency stays 0;
   run direct=False and assert it rises (the positive control, showing the test can fail). Stronger, but
   it skips exactly where CI runs, so it cannot be the only guard.
3. **The fd's flags**: assert the fd opened by the reader carries O_DIRECT (`/proc/self/fdinfo/<fd>`,
   bit `040000`). `test_uring_file_reader.py` does not check it.

### 4. `test_draft_prefix_selects_mtp_experts`, `test_exl3_expert_layout.py:151` `[re-verified]`

The guarded invariant: `prefix="mtp"` selects the draft experts' offsets, not the target's. The fixture
builds the mtp tensors as renamed copies of `_expert(0, 0)`, so they carry the same `(0, 0)` key as the
`layers.` row, and the only assertion is `list(layout.records) == [(0, 0)]`. No path or offset assertion,
though the two rows sit at different offsets.

**Mutant missed:** `re.escape(prefix)` -> the literal `"layers"` at `exl3_expert_layout.py:64`. The test
stays green while a draft caller would read the **target model's** expert bytes: silently wrong weights.

**Reach, stated precisely.** Today no production caller passes a prefix (every call in `python/` and
`scripts/` uses the default), and the DSpark draft path (`DSV41_REFERENCE.md` section 14.3: draft weights
are the `mtp.*` tensors inside the main EXL3 checkpoint) does not run on the EXL3 stack yet. So this is a
latent hazard, not a present corruption: it would start when the draft path is wired to `prefix="mtp"`.
The only check that would catch the mutant is a manual test, `test/manual/dsv41/test_exl3_checkpoint_layout.py:27`,
which needs the real checkpoint and asserts shapes (`(3, 128)`, `row_bytes == 17_739_276`), not offsets.
The CI-side guard cannot fail, and the guard that could is not in CI.

**Repair sketch:** put the mtp tensors at expert ids distinct from the target's, or make the two sets differ
in content, and assert the record's path, `file_offset` or row bytes belong to the mtp tensors.

## Tier 2: control flow and policy

### 5. `TestInsertOnMiss.test_victim_is_the_lowest_score_resident_not_routed_in_that_forward`, `test_expert_residency_gpu.py:811` `[reviewer-established; arithmetic matches the code I read]`

Guarded invariant: a resident routed in the previous forward is never chosen as the insert-on-miss
victim, which is the property that keeps a slot from being recycled under the forward using it.
`IOM_DECAY = 0.98` (test :563), and production folds `insert_scores * decay + route_counts` before
ranking (`expert_residency_gpu.py:510-511`). With the test's scores (`lowest = 1.0`, `second = 2.0`,
`lowest` routed once), `lowest` becomes 1.98 and `second` 1.96, so **`second` is the lowest score whether or
not routed residents are excluded**. For any decay below 1, `d + 1 > 2d`.

**Mutant missed:** delete `& ~routed.gather(1, slot_experts)` at `expert_residency_gpu.py:614`. The victim
is still `second` and every assertion passes. The DIRECT twin at `:1378-1409` is sound (a neutral warm step
and 1.0 against 50.0). The randomized test at `:779` compares against a reference that does exclude routed
residents, so it probably catches the mutant, but that was not proven. This class is the GPU residency
updater, which EXL3 does not use today (section 2.2 of `PROMOTION_ASYNC.md`), and the test runs only on a
CUDA runner.

### 6. `test_invalid_reassignment_does_not_change_residency`, `test_expert_hot_cache.py:483` `[re-verified]`

**Shape worth naming: an assertion downstream of the property, where the property has no observable effect
on what is asserted.** The only post-failure check is `self.assert_routes([[3, 7]])`, which compares
*gathered bytes* (`:191-204`). Those bytes are identical whether experts 3 and 7 are still resident hits or
have become misses that fetch the same rows. Nothing asserts `cache.slot_to_expert`, `cache.slot_states` or
`hot_hit_rows`. **Mutant missed:** move the `len(set(desired)) != len(desired)` check
(`expert_hot_cache.py:653`) below the retire loop (:675-677); `reassign([3, 3])` then retires 7, raises,
and leaves 7 a miss. The bytes are still correct and the test is green.

### 7. `test_a_reserved_or_unusable_core_is_refused`, `test_exl3_ram_miss_thread.py:283`, the assertion `assert not host.threaded` `[reviewer-established]`

`self.threaded = False` in `__init__`, set True only after `start_thread` returns
(`exl3_ram_miss.py:442`, `:481-482`). The call raises for both cases, so the flag is False regardless of
C++. **Mutant missed:** delete `tier_->set_threaded(false)` at `exl3_ram_miss_host.cpp:2473`. Repair:
assert `host.counters()["running"] == 0` or that `host.pump()` still works.

### 8. `test_the_host_source_agrees_with_the_lease_layout_once_it_defines_it`, `test_exl3_ram_miss_device_args.py:254` `[reviewer-established; known]`

Your item 5. It skips until the host defines a `kLease*` constant, and today the greps match nothing, so it
cannot fail. The guard logic itself is sound (`test_the_host_lease_guard_can_fail` at :263 exercises every
failure branch on synthetic input), so it will bite once the constants exist. It is a skip-until-later stub,
which is not a defect while that is visible.

### 9. The doorbell knob tests, `test_expert_doorbell_copier.py:285-310`, `:342-353` `[re-verified]`

`test_poll_modes_match_reference`, `test_head_stores_match_reference`, `test_overlap_preference_matches_reference`
assert only `timeouts == 0` and the byte reference. `_COUNTERS`/`_STATE_WORDS` (`expert_doorbell.py:69-88`)
have no entry for `poll_mode`, `head_store` or `prefer_overlap`, so nothing observable says the knob arrived.
**Mutant missed:** `self.poll_mode = _POLL_MODES[poll_mode]` becomes `= 0` (`expert_doorbell.py:300`), and
the same for `head_store` (`:301`) and `prefer_overlap` (`:350`). All parametrized cases stay green, and the
volatile, noncoherent and no-overlap variants are never exercised. Contrast the test at `:206-208`, which
reads `copy_api` back through `stats()`.

### 10. Refusal tests that two independent guards answer `[reviewer-established]`

Shape: two guards refuse the same launch, and the `match=` string is satisfied by either, so deleting one
guard leaves the test green. No unsafe launch starts, because the other guard refuses, so the risk is only
that a guard can be removed unseen:
- `test_expert_stream_requirements_exl3.py:191` `GRAPH_GATHER`, `:193` `PINNED_HOST_MB`, `:195`
  `HOT_GPU_MB`: each matches one guard in `memory_hook.py` and a second in `expert_stream_requirements*.py`.
- `test_expert_residency_gpu.py:776` `match="decay"` matches both `expert_hot_cache.py:1046-1049` and
  `expert_residency_gpu.py:207-208`.
- `test_nvfp4_expert_offload.py:1084` (graph gather plus prefetch): deleting `memory_hook.py:169-172` leaves
  the test passing, because the pinned-cache guard at `:218-219` raises a message that also contains
  "expert prefetch". The combination is unreachable anyway.

## Tier 3: observability and trivia (listed so they can be ignored deliberately)

| test | what cannot be seen | status |
|---|---|---|
| `test_exl3_ram_miss_stage_trace_causal.py:111` resubmit keeps first submit | nothing compares an extent's submit stamp to the record's, so first and last submit are indistinguishable; missed mutant `host.cpp:906-911` | `[re-verified]` |
| `test_exl3_ram_miss_stage_trace.py:109` per-drive bytes sum to the total | the fixture has one drive (`roots=()`, and the mirror roots share one `st_dev`), so the sums are trivially true; missed mutants `host.cpp:519-522`, `:1049`, `:1200` | `[re-verified]` |
| `test_exl3_verify_expert_mirror.py:184` first byte is lowest by row offset | separates "first name wins" from "lowest offset", not "last name wins"; no verdict effect | `[reviewer-established]` |
| `test_expert_doorbell_copier.py:1265` `assert prime_row["bytes"] == 0 or prime_row["enqueued_ns"] != 0` | always true; the rest of that test is real | `[reviewer-established]` |
| `test_expert_hot_cache.py:669` copy backend "before initial population" | observes only the end state | `[reviewer-established]` |
| `test_expert_gather_experts.py:164` validation once up front | the spy counts dispatches, not validations | `[reviewer-established]` |
| `test_expert_gpu_pull.py:239` forking without a join | asserts an absence; a demonstration test, the positive property is covered at `:226` and `:73` | `[reviewer-established]` |
| `test_exl3_row_reader.py:253` roots that are the same file share one id | depends only on the fake reader's own dedup | `[reviewer-established]` |
| `test_exl3_expert_layout.py:94` header drops metadata | the fixture never writes `__metadata__` | `[reviewer-established]` |
| `test_expert_prefetch_pricing.py:105` join by forward | identity mapping fixture | `[reviewer-established]` |
| `test_expert_prediction_predictors.py:81` decayed counts | one observe on zero counts, so `mul_(decay)` is invisible | `[reviewer-established]` |
| `PROMOTION_ASYNC` section 11.2 M0 test | fails on old code only because a list has no `.tolist()` (see `EXPERT_ID_ROUNDTRIP.md`) | outside this scope; known |

## Known items and where this sweep landed on them

| known item | in this sweep's scope | result |
|---|---|---|
| `pack_one`'s `filled >= needed` coverage | yes (`test_exl3_ram_miss_split.py`) | the reviewer found the split tests sound: the coverage check and the stale-CQE, EOF and generation-wrap tests each fail if the guarded line is mutated, and the file's fault firing is confirmed by extra CQE counts. Its remaining note is that two tests overlap with a second defence (post-loop `clean` check, `host.cpp:633-636`) so deleting one guard alone would not show |
| the generation gate (`GENERATION unknown` then `VALID`) | no (a script and manifest, not these suites) | not covered |
| `PROMOTION_ASYNC` M0 test | no (a design document) | known, in Tier 3 |
| `test_a_failed_read_publishes_none...` | yes | confirmed, and finding 1 adds the mutant |
| the `kLease*` agreement test | yes | confirmed as a skip-until-later stub, finding 8 |

## Coverage plumbing: a separate class, not counted above

These are not tests that pass for the wrong reason; they are tests the registered suite runner does not
run. Counted with the repo's own parser (`ut_parse_one_file`) over the 82 in-scope files:

- **22 files (279 tests) have no `register_*_ci` call**, among them `test_expert_transfer.py`,
  `test_expert_doorbell_copier.py`, `test_expert_cache_transfer_warp_geometry.py`, `test_expert_residency*.py`,
  the `test_expert_prediction_*` files and `test_nvfp4_expert_offload.py`.
- **17 files (171 tests) are registered but have no `if __name__ == "__main__":` entry**, among them
  `test_exl3_read_split.py`, `test_exl3_lease_block.py`, `test_uring_file_reader.py`,
  `test_expert_graph_gather.py` and the stage-trace tests.
- `collect_tests` raises loudly for both (`No CI registry found`; `missing if __name__ == "__main__"`), and
  `scripts/lint/check_registered_tests.py` is a pre-commit hook. So this is loud, not a silent green. It
  means those 39 files run only when someone invokes pytest by hand, which is how this team runs them. If
  anyone relies on the suite runner for them, they are not covered.
- Related and informational: most GPU classes are `skipUnless(torch.cuda.is_available())` under
  `register_cuda_ci`, so a broken CUDA runner reports skips, not failures. Several byte tests skip where
  O_DIRECT is unavailable (tmpfs), and three permission tests skip as root.

## Mutation tests worth spending time on, in order

1. `exl3_ram_miss_host.cpp:2054` (finding 1), and the same edit as Task 6 V2 intends.
2. `expert_hot_cache.py:292`, delete the generation comparison (finding 2).
3. `verify_expert_mirror.py:519` and `:534` separately, `direct=False` (finding 3).
4. `exl3_expert_layout.py:64`, `re.escape(prefix)` -> `"layers"` (finding 4).
5. `expert_residency_gpu.py:614`, delete the routed exclusion (finding 5).
6. Weak, unproven, byte-relevant (reviewer lists):
   - `test_exl3_ram_miss_thread.py:110` never asserts an advisory ran (skips may consume them all): mutant where the service ignores `pause_requested_`.
   - `thread.py:465` and `:450` may have an empty resident set: assert cancelled or skipped counts.
   - `test_exl3_read_split.py` random property checks sum, alignment and count only; the pinned-tuple tests catch a remainder change, and they do.
   - `test_file_row_reader.py`: nothing asserts that direct plus an aligned destination really uses the O_DIRECT file (forcing `use_direct = False` at `file_row_reader.py:158` stays green).
   - `test_expert_hot_cache.py:181`: the four file tensors are byte-identical, so swapping tensor names in `_prepare_promotion` is invisible.
   - `test_expert_stream.py:355-358`: the slot-uniqueness loop is guarded by `if len(call.args) == 3` and may be empty.
   - `test_expert_graph_gather.py:846-852` compares `[] == []` if the profiler returns nothing.
   - `test_expert_pinned_graph_gather_cuda.py:53-64`: slot equals id, so slot mapping and id mapping are indistinguishable.
   - `test_expert_residency_batched.py:180` asserts only `promotions > 0`.

## What was not read

- Slice E did not read the nine `test_exl3_*_gpu.py` and `_cuda.py` files, `test_engram_parity.py`,
  `test_engram_row_cache_direct.py`, the `test_ref_*` files, `test_exl3_vs_fp8_engram_wkv.py`, or most of
  `test_graph_parity.py` (about 1,900 lines). Those are GPU-bound or outside the byte-safety core.
- Slice D skimmed about 20 of its 236 tests rather than tracing them line by line.
- Reviewers read the fixtures and the production paths a test hits, not every production line. The C++
  thread and `sim_*` paths in `test_exl3_ram_miss_service.py` were not verified beyond their Python
  assertions.
- No mutant was run for any finding. Every "would stay green" is a source argument with a named line.
- This sweep found what it found in about 1,140 tests. It says nothing about the suites it did not cover,
  and a clean file here means "the reviewer could not construct a surviving mutant", not "no surviving
  mutant exists".
