# Review of the storage v2 plan's Task 6 section against the documents it summarises (2026-09-21)

Reviewed: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, "Task 6", lines 232 to about 550 at HEAD
`e1c7bde9e6`. Sources read: `PER_ROW_TRANSFER.md` (current), `PER_ROW_PRECHECK_REVIEW.md` (R1 to R6, by `t3-topology`),
`PER_ROW_PRECHECK_PREREG.txt` and `per_row_precheck_result.txt`, `per_row_precheck.py`, `PER_ROW_TRANSFER_REVIEW.md` (mine), and the
source (`exl3_ram_miss_host.cpp`, `exl3_ram_miss.cuh`). Read-only: no GPU, no traces re-read (that is drive I/O on a contended box),
nothing run except `grep` and `git`. Line references are to the plan file at that HEAD and **drift by a few lines as the plan is edited; the quoted text is the locator**.

**Which sources are mine, so the weighting is visible.** The plan's transcriptions of my first and second reviews (F1 bound 33.5 vs 38.6,
the same-`mutex_`-section point, the V2 cleanup landmine, `kBusySeq`, the safety-window refinement, the G3 expiry correction) are my own
findings: I am checking the lead's transcription of my work. R1 to R6 (recompute, FIFO completion, A1 bounds, the 511-step correction) are
`t3-topology`'s and the `kBusySeq` window figures and the 3-slot demonstration are `t1-instrument`'s: on those I am neutral.

**Severity.** HIGH: a reader who trusts the section builds the wrong thing. MEDIUM: two passages contradict each other, or a stated fact
is wrong. LOW: precision.

## Verdict

**The section is not internally consistent, and the inconsistency is structural rather than a few slips.** It is built as an audit trail
(a summary, then a chronology of corrections in place), and the summary and the earliest paragraphs were not brought forward as later
corrections landed. Three claims that the later text supersedes are still stated flatly in the summary or in earlier paragraphs. The
figures are applied correctly where the R6 correction was applied, but the correction was applied to one paragraph and not the ones around
it. Fidelity to the sources is good: I found no invented finding, and the transcriptions of R1 to R6 are accurate, with the two exceptions
in section D. No HIGH finding; five MEDIUM.

## A. Passages that contradict each other or describe a superseded state

**A1. MEDIUM: the SUMMARY ("SUMMARY OF THIS TASK'S STANDING") is stale against the body in four ways.** It is the thing a reader will stop at.

- Item 5, "Every figure below is conditional on (3) and on assumption A1, which the traces cannot check": the body then says "**A1 no longer
  has to be assumed**" (line 342) and that the data bound the hit-lane placement. Two statements about A1 in one section, opposite.
- Item 3, "That signal is also a *safety* requirement, not only a performance one": the body's own "Refinement of the safety claim above" says
  an early copy needs "*either* a service ownership grant *or* a gate", and V1b (below) gates without a lease. The summary keeps the unrefined form.
- The summary never mentions V1b or `kBusySeq`, which is the largest single change to the plan's costing ("a cheaper V1 exists"), nor the
  withdrawal of "fails loudly", nor the FIFO-completion attribution, nor the R6 figure correction. A reader of the summary alone has none of the
  four.
- Item 1's "about 14 ms/step worse" is a pre-R6 figure (-14.35); the R6 paragraph in the same section gives -13.08 (see B1).

**A2. MEDIUM: the BLOCKING DEPENDENCY block (lines 351-364) is false for V1b as written.** "Every figure in this blockquote is conditional on
building `[REQ 1]`" and "It binds V1 two-phase exactly as hard as V2 per-row" were written before `kBusySeq`. Lines 402-412 then state that V1b
gets the 87% with "**no service change at all**", i.e. without `[REQ 1]`. Both cannot hold. What holds: `[REQ 1]` binds the *lease-based, durable*
V1 and V2; V1b avoids it and borrows the temporal exclusion instead. The block needs "for a Task-5-compliant mechanism".

**A3. MEDIUM: the Task 8 expiry claim is corrected, then re-asserted.** Lines 415-419 correct it ("I first wrote that it expires when the service
becomes asynchronous, 'which Task 8 exists to make it'. That is wrong ... it expires when the service stops serving one request at a time"). The next
paragraph (lines 420-424) says "a mechanism whose correctness argument evaporates when the service goes async is a liability in a plan whose **Task 8
makes it async**". That is the withdrawn claim. Either the sentence is wrong or the correction is; by `PROMOTION_ASYNC.md` M2 (promotion admission as a
class on the same thread, below demand) the correction is right, so the sentence should read "a plan whose Task 5 wording contemplates an asynchronous
`progress()`".

**A4. MEDIUM: the sequencing consequence contradicts V1b (lines 459-462).** "Task 6 cannot be accepted before Task 5 (b) lands ... This orders the
remaining plan work **regardless of which mechanism is chosen**." V1b is a mechanism that needs no Task 5 (b) by the plan's own account (lines 402-412).
The claim is true of the lease-based mechanisms and false of V1b; if the plan intends that V1b is not an acceptable outcome for Task 6, it should say so
(it says "the lease is still what this plan wants", which is a preference, not a rule) and say why the gate would refuse it.

**A5. LOW-MEDIUM: "unverified" and "no document states" labels the later text answers.**

- Lines 414-417, "Two things unverified: whether the device reliably observes `kBusySeq` before it clears, and whether every planned lane is in
  `protect` for every producer." Lines 486-519 then answer both: observation is "NOT guaranteed, but fail-safe" with measured windows, and a planned lane
  outside `protect` is shown to be evicted silently (the 3-slot demonstration). The label should be updated, not left as an open question.
- Lines 486-490, "Latent and not closed: a planned expert absent from `protect` ... failing loudly", sits above the WITHDRAWN paragraph (line ~491) that
  says "fails loudly" does not hold for the early-read variant. The first is true only of today's path; it should carry "(today's path; see the
  withdrawal below)".
- Lines 518-520, "neither of which any document states as load-bearing": the plan itself states both invariants as load-bearing at lines 407-412
  ("safe only under today's invariants -- one service thread ...; `take_slot_locked` never evicts `wanted`; eager paused"), and `PER_ROW_TRANSFER.md`
  section 3.3 and my review section A say so too. "No document" is wrong unless it means "no document that predates this task"; it should say that.
- Lines 515-516, "a hit copy lasts **milliseconds**, far longer than the window itself": the window is 5.03 ms (mirrors on) or 10.25 ms (off) for
  requests that read rows (line 508), which is *longer* than a hit copy (about 2.3 ms for two lanes at 1.055 ms). "Far longer" is true only of the
  0.6-1.8 us window of requests that read nothing. As written it reads as general and contradicts the number two lines above it.

**A6. LOW-MEDIUM: "the whole of V1's protocol delta" (line 439).** The cost split says (a) publish hit lanes' `RowResult` inside the reservation section is
"the whole of V1's protocol delta". `PER_ROW_TRANSFER.md` section 3.1's own table lists, for V1, a stage wait that polls the hit lanes' words, one more copy
stage, and "a partial terminal mask (needed only if stage 2 fails after stage 1 copied)"; `lease_model.py` does not model per-lane copy or a partial mask
(design 18.3). "Whole of the *service-side* delta" would be accurate.

**A7. LOW: the opening blockquote says two-phase has "no partial terminal mask" (lines 300-302).** Same source: `PER_ROW_TRANSFER.md` 3.1 says V1 needs one when
stage 2 fails after stage 1 copied. "No per-lane partial mask" or "a simpler mask" would be true.

**A8. LOW: the task has no checklist or gate for the mechanism it chose.** Lines 522 and following choose two-phase, but every checkbox and the pseudo-code
after it (lines 528-560) specify V2, "retained as the specification of that variant", and the Gate paragraph is unchanged. A reader implementing Task 6 finds
no V1 checklist. The design's `[REQ 5]` suggested retitling; the section heading still reads "Start per-row GPU transfers".

## B. Figures

**B1. MEDIUM: pre-R6 and post-R6 figures stand side by side, and the R6 warning does not cover the ones above it.** The R6 paragraph (line 328) says
"EVERY ABSOLUTE FIGURE **BELOW** IS 3-7% HIGH". Its scope is what follows it. The figures *above* it are pre-R6 and unmarked, and several figures *below* it
are pre-R6 too:

| Where | Stated (pre-R6) | Post-R6 (R6 table) |
|---|---|---|
| Summary item 1 (line 269) | about 14 ms worse | 13.08 |
| Opening blockquote (287-288) | 19.2 ms/step, about 7.1% after launch cost | 17.79; 6.5% (the plan's own line 336 gives 6.5%) |
| Opening blockquote (291) | at most 5.06 ms, about 2.0% gross | 4.93 (simulation) or 4.90 (arithmetic over 511); about 1.9% |
| A1 paragraph (343-349) | 111.9 hit lanes, credited 31.9 | 108.4 and 29.3 (R6); marked "not recomputed", correctly |
| "Two corrections" (446-448) | V1's own bound is 33.5 ms, not 38.6 | 30.88, not 35.80 |
| "Chosen first mechanism" (line 522) | "14 ms/step worse than V1 (38.6 - 5.06 - 19.2)" | 13.08 (35.80 - 4.93 - 17.79) |

The R6 paragraph's own figures are the correct ones and match `PER_ROW_PRECHECK_REVIEW.md` R6 exactly (BEST 35.80, RANDOM 17.79, two-phase 30.88,
+4.93 / -13.08, hit lanes 31.9 to 29.3, `(RANDOM - 1) / T` = 6.5% against 257.5 ms). **The error is placement, not arithmetic:** the correction paragraph
was inserted in the middle, and the paragraphs before it and the "Chosen first mechanism" paragraph after it kept the old numbers. The verdicts do not
change (R6 says so and I agree), but the same quantity appears with two values in one section (7.1% and 6.5%; 33.5 and 30.88; 14 and 13.08).

**B2. LOW: "3-7% high" is not what the R6 table shows.** Registered to corrected: BEST 38.61 to 35.80 (7.3%), RANDOM 19.17 to 17.79 (7.2%), two-phase 33.53 to
30.88 (7.9%), per-row over two-phase best 5.09 to 4.93 (3.1%), random 14.35 to 13.08 (8.8%), hit lanes 31.9 to 29.3 (8.2%). So the range is about 3% to 9%
depending on the quantity, not 3-7%. The "3-7%" is copied from `PER_ROW_PRECHECK_REVIEW.md` R6's prose, whose own table disagrees with it. (The plan's
"3-7%" also appears in the commit message and in the paragraph title.)

**B3. LOW: the "87/13" split is 86/14 after R6.** two-phase / BEST = 30.88 / 35.80 = 86.3%; miss-only / RANDOM = 2.46 / 17.79 = 13.8%. R6 says the ratio "stays
about 0.13"; it is 0.138. Immaterial to the verdict (the SPREAD-IRRELEVANT bar is 0.25) but the plan states 87/13 in several places.

**B4. LOW: "511 decode steps" carries an unexplained 3-step gap that the plan omits.** R6: "the client saw 4 x 127 = 508; I do not know the 3-step gap."
The plan states 511 as fact (line 331). The effect on any figure is 0.6%, but the caveat travelled with the number in the source and did not in the plan.

**B5. LOW: "The `495` is in `per_row_precheck.py` itself" (line 338) is mis-transcribed.** There is no literal 495 in the script. It divides by
`n_steps = len(used_steps)` (line 136), the number of distinct `graph_step` *lines* used, which is 495 on this trace. The bug is real (it treats a line as a
step, so 16 merged lines are undercounted) and "the pre-registered script carries the bug" is right; "the 495 is in the script" is not.

**B6. LOW: the 6.5% and the step-time denominator.** `(17.79 - 1) / 257.5 = 6.5%` uses the 257.5 ms denominator, which R5 says includes the INVALID `task1-6` arm
and the disturbed `task1c-3`; the clean value is 254.4 ms (6.6%). The design document already moved to 254.4. Immaterial to every verdict, but the plan and
the design now quote different step times for the same shares.

**B7. LOW: "about 1% net of launch cost" versus the design's "1.1-1.5%".** Lines 292-293 and 522 say per-row over two-phase falls to "about 1%" net of launch
cost; `PER_ROW_TRANSFER.md` 0 and 5.6 give 1.1% to 1.5% for the best order (g of 8-14 us per stage triple, unmeasured). "About 1%" is the low end presented as
the estimate, against a resolution the plan also calls optimistic.

## C. Fidelity to the sources: what checks out

Checked and correct in the plan:

- R6's numbers (BEST 35.80, RANDOM 17.79, two-phase 30.88, +4.93 / -13.08, hit lanes 29.3), the 16 merged lines with `steps` = 2 and `routed_rows` 480 against
  240, `k` about 3.3 against the capped 6 on about 6% of requests, and "no verdict changes" with `(RANDOM - 1) / T` = 6.5%.
- R3's FIFO finding: 0 inversions in 1,842 m>=2 requests on each of three mirrors-on arms and the mirrors-off arm; 0 of 4,750 consecutive rows on one drive
  part; same-reap ties 3.7% of adjacent pairs mirrors-on, 0% mirrors-off; "`MIRROR_ROWS`' tail means bunching, not reordering".
- R2's bounds: 10.8 and 63.6 of 111.9 hit lanes, worst-placement two-phase about 4%, random-order per-row about 2.8%, floors at f = 0.26 and 0.145, and the
  plan's caveat that they were not recomputed at 511.
- R1: 2.55 is the registered `RANDOM-miss-only` and not V2 minus V1; "at most 5.06, 2.0% gross, above 1.5%".
- R5: the 1.5% resolution "is itself optimistic" because `task1e` is UNRESOLVED.
- Source facts I verified in the C++ and CUDA: `kDemandDone` is stored in one place (`pump_demand`, after `handle_demand`); the wait kernel polls only it;
  `kBusySeq` is stored first in `handle_demand`, cleared after `set_status`, before `demand_done`; advisories never store it; `serve()` reserves under `mutex_`
  and drops it before `read()`; `take_slot_locked(..., wanted, false, ...)` never evicts `wanted` and evicts LRU non-`hot` non-`wanted` rows; schema 4 (`4e63616666`)
  records `lanes`.
- The "WITHDRAWN" logic is correct: with the map read early, a planned hit outside `protect` can be evicted by the request's own reservation and is never re-read;
  planned lanes are VRAM misses, so `hot` does not protect them.

I did not re-derive R1 to R6 from the traces (that is drive I/O; `t3-topology` ran it), and I have not verified the `kBusySeq` observation windows (5.03 / 10.25 ms,
0.6-1.8 us, the 0.2-0.9 us gap) or the 3-slot demonstration: **they are cited to a message id (`58f2ccd5`) and I could find no committed artefact for them.** A
future reader cannot re-check them from the tree. They are plausible against the source (the store and the clear are adjacent to `serve()`, and the read-request
minima match the read-wait p10 of 5.2 ms) but they are unverifiable as cited.

## D. Attribution and overclaiming

**D1. LOW: the pack-order attribution was also mine.** The plan ("**Correction:** the perfect pack-order result") says the tie-break attribution was made "in earlier revisions of this plan -- by
me". It was also in my review (section E item 2: "I agree with the document that it is the tie-break", and F5). R3 refutes it, and the plan's correction is right;
my record is corrected by R3 too and I withdraw it. The plan may say "by me and by the first review".

**D2. LOW: "The ordering comes from per-drive FIFO completion" is an inference stated as a finding.** R3 measures that completion stamps are in ordinal order (0
inversions) and that the tie-break decides only 3.7% of pairs. That the *cause* is drive FIFO (rather than uniform-size reads submitted in row order, queue depth, or the
mirror split's structure) is the natural explanation and is not separately tested; NVMe does not guarantee FIFO completion. The plan does carry it as an explicit
assumption ("must list FIFO completion per drive part as an explicit assumption") which is the right handling; "comes from" is one word stronger than the evidence.

**D3. MEDIUM-LOW: "It is justified" (summary item 2).** "Two-phase ... is the chosen mechanism. It is justified." The plan's own text says the saving is conditional on an
unbuilt signal (item 3), on an unmeasured lease-mode baseline cost X (line 524, "must be netted out or any Task 6 arm flatters itself"), and on a link that early
copies cannot beat (my G1, not yet in the plan). "Supported by the modelled ceiling as an upper bound; not yet measured" is what the evidence licenses. The summary
labels are otherwise well calibrated (CONFIRMED where checked at the source, WITHDRAWN, "not verified"), and this is the one place a label is stronger than its basis.

**D4. Not in the plan yet, and belongs in the summary:** the link bound (`PER_ROW_TRANSFER_REVIEW.md` G1): `c` is the Gen3 link's time per row, so early copying moves link
time into the read wait and the hit-phase saving is capped by link idle inside read waits; V1's ceiling is a ceiling by construction; promotion copies on the same link subtract.
The lead asked me to write it up and said it would be recorded; I did not find it in the section.

## E. What I did not check

The R1 to R6 recomputation against the raw traces; `kBusySeq` window figures and the 3-slot demonstration (no committed artefact); whether `PROMOTION_ASYNC.md` M2's
"promotion admission cancelled at the next row" matches the eventual C++ (it is a design); Task 5's own text about "asynchronous `progress()`" (I cited it as
wording the plan should reference, from the plan's Task 5 section, not re-read in full); the plan outside Task 6 (the coverage table's Task 6 row, Task 9's readiness-aware
gather row); anything the `task1f` arm will produce. Nothing was run.

---

# Second audit: the rewrite (`7990275c98`, `1556a30a56`), 2026-09-21

Same terms and limits as above: read-only, no traces re-read, nothing run. The plan's Task 6 section is now lines 262 to 652 (390 lines, of which
about 300 are one blockquote). Line references are to the plan at `1556a30a56`; quoted text is the locator.

## Verdict

The rewrite fixed what the first audit named and introduced three new problems of the same families. **The most important is that the rule written
for the V1b exclusion is not enforced by the Gate it points at (F1, HIGH).** The audit trail is not history in the sense its label promises (F3).
Fidelity of the new figures to `t3-topology`'s table is good.

## Findings

### F1. HIGH: "The Gate below refuses a mechanism whose correctness rests on either" is false; the Gate is unchanged

The sequencing paragraph ends: "**V1b is nonetheless excluded**, and as a rule rather than a preference ... The Gate below refuses a mechanism whose
correctness rests on either." The Gate (line 652) reads, in full: "Both demonstrated I/O/H2D overlap and untraced end-to-end benefit at unchanged cache
capacity. If launch/head-of-line cost cancels the benefit, retain Task 4 and record Task 6 as rejected; evaluate a readiness-aware gather or native DMA in
Task 9 instead of claiming success." It says nothing about borrowed invariants or a `planned subset of protect` assertion. **A V1b that shows overlap and a
benefit passes it.** The checklist above the Gate (V2's) has no line for either. So the plan states a prohibition that its acceptance test does not
implement, which is what you asked me to check for, and it is worse than the preference the first audit found because it reads as enforced.

The reasons for the rule are also weaker than a rule needs:

- **Reason 1 cancels itself.** "Its safety is borrowed from invariants no document predating this task states as load-bearing." By the plan's own text the
  invariants are now stated as load-bearing (by this plan, `PER_ROW_TRANSFER.md` 3.3 and my review section A). The absence being cited has been remedied
  by this task, so under the criterion as written V1b is no longer excluded on this ground. The real ground is durability: the borrowed exclusion expires
  when the service stops serving one request at a time, and Task 5's wording contemplates an asynchronous `progress()`. The plan states that ground elsewhere
  and should use it.
- **Reason 2 overstates.** "A device-side `planned subset of protect` assertion that **no producer guarantees**." The plan's own hint analysis says that for the
  router-miss producer `protect` is filled from the routed experts by `_apply_graph` and every planned lane is inside it. What is absent is a device-side
  *assertion* of it and a guarantee for a *future* producer. The demonstration (a request protecting only expert 4 while lane 0 is planned) constructs a
  producer that does not exist.
- **The summary cross-reference is wrong.** Item 6 says V1b "is excluded, and item 4 says why". Item 4 is about Task 5 (b). Item 3 mentions the device-side
  invariant; nothing in the summary states an exclusion rule.

**What would make it true:** a sentence in the Gate itself (for example "a mechanism is refused if its safety depends on a gate that is not a lease, or if it does
not assert `planned subset of protect` on the device") and, in the checklist, the assertion as a checkbox with a mutation control (a producer whose planned lane is
outside `protect`; the test must show it is caught).

### F2. MEDIUM: "the second [unverified item] *false*" is stronger than its basis

"Two things unverified: ... whether every planned lane is in `protect` for every producer. **Both are answered below** -- the first NOT guaranteed but fail-safe,
the second *false*, and false in a way that yields silent wrong bytes." The demonstration shows that *if* a planned lane is outside `protect`, an early copy can
read a slot the request then evicts, silently. It does not show that any producer puts a planned lane outside `protect`; the plan says elsewhere that no
production producer other than `expert_row_plan.py`'s does. The accurate answer is "true for the only production producer; not guaranteed by construction for a future
one; the failure, if it occurred, would be silent under early read." "*False*" is the wrong label and it feeds F1's reason 2.

### F3. MEDIUM: the "audit trail ... must not be quoted" is not history; it holds the live specification, and it mixes updated and stale figures

The CURRENT FIGURES block says the trail below "records how [the figures] moved and **must not be quoted**". Checked against what is actually below it:

- **The trail contains the only statement of several live things.** V1b is defined there and only there (summary item 6 points at it); the `kBusySeq` facts a
  consumer must honour (clears before `demand_done`; advisories never set it; poll both words); the safety invariants (a) and (b); the `planned subset of protect`
  demonstration; the V2 cleanup landmine; the same-`mutex_`-section correction; the hint analysis; the unarmed-record analysis. None is history. A reader who obeys
  "do not quote" loses them; a reader who ignores it quotes obsolete numbers.
- **Some trail paragraphs were updated to current figures, some carry "(pre-measurement)" parentheticals, some are marked history, some were left.** The paragraph
  after the first `---` uses 20.07, 7.4%, 4.93, 1.9% (current) but says "88% ... and only **13%**" in one sentence (88 + 13 is not 100; the table says 88/12) and "about 1%
  only net of launch cost" (the "Chosen first mechanism" paragraph says 1.1-1.5% and records that "about 1%" was the earlier revision). BLOCKING DEPENDENCY opens with
  "The 7.5-15% saving, and the 87% share" (old) and, three lines later, "V1's advantage is the 88% hit-lane share" (new). "Two corrections from the first independent review"
  carries "33.5 ms, not 38.6" with "(Both are pre-measurement; the current pair is 35.74 against 40.67.)". So which figure in the trail is live cannot be told from position.
- **A stale sentence in an updated paragraph.** "The caveat that carries the result is that hit lanes are assumed to spread evenly over layers; the traces do not record lanes
  per layer ... a schema-4 trace can replace the bounds below with a measurement." The measurement has landed; the summary says "A1 no longer has to be assumed"; A1's layer-level premise
  held and its per-request count did not. This sentence still describes the state before the trace.
- **Embedded history in live text.** "An earlier revision of this sentence said ...", "An earlier revision quoted 'about 1%'", "As first written this bullet said ..." appear inside
  the paragraphs a reader will quote. They are useful once and noise afterwards.

**Answer to your question 2: yes, the section would be more honest shorter, and I would delete the history rather than mark it.** A marked chronology is a weaker device than
deletion for exactly the reason above: "must not be quoted" is a request, and the reader cannot tell live paragraphs from history by looking. The plan is not the place for the record of
how each number moved; that record exists (`PER_ROW_TRANSFER.md`, `PER_ROW_TRANSFER_REVIEW.md`, `PER_ROW_PRECHECK_REVIEW.md`, this file, and the commit messages, several of which
already say what each correction was). A structure that survives future edits: (i) a short standing block (the summary, the table, the two rules); (ii) live technical notes, each written
once in current-state form, with no "an earlier revision" clauses (V1b and its invariants, the `kBusySeq` facts, the hint analysis, the V2 landmine, the cost split); (iii) a single
sentence pointing to the documents that hold the history. That is about a third of the current length and every sentence in it is quotable. The evidence that the current shape
decays is in this section: the rewrite that introduced the table also introduced two of the mixed paragraphs above.

### F4. MEDIUM-LOW: the new table's "two-phase" row is the ceiling by construction, and the table does not say so

Measured hit lanes are 33.9 per step; 33.9 x 1.055 ms = 35.77 ms, which is the two-phase figure (35.74) to within 0.1%. So two-phase is not a model output with an independent
spread of assumptions; it is "every hit lane's copy hides completely", i.e. hit lanes x c, and it can only be lower if some read waits are shorter than `h*c`. Summary item 7
says "V1's ceiling is a ceiling by construction" and the block says the savings are modelled; a one-line note in the table ("two-phase = 33.9 x 1.055, every hit copy hidden") would make the
reader see that the only inputs beyond the measured `k` are `c` and A2, which is the honest state and reduces the chance the 13.9% is read as a prediction.

### F5. LOW: smaller items

- **The 257.5 ms denominator is recorded, not reconciled.** The block says the plan and the design "quote different step times for the same shares and should be reconciled" and keeps
  257.5. Whether that is the reconciliation the lead described or a note that one is owed, a reader takes the shares as stated; at 254.4 they are 1.2% higher (7.5% rather than 7.4%).
  Immaterial; say which one governs.
- **"A1 was first bounded, then measured, and the prediction held."** The bounds held. The A1 *estimate* (29.3 / 31.9) undercounted by about 14% and its per-request `k` was exact for
  36% of requests. "The prediction held" is true of the bounds and reads as true of A1. Say "the bounds held".
- **The "3.0x the registered SUPPORT floor"**: the floor was 11.7 (37% of 31.9) in the first document and 10.8 rescaled; 33.9 / 11.7 = 2.9 and 33.9 / 10.8 = 3.1. "About 3x" is right; "3.0x" implies a
  precision the floor does not have.
- **The 88/12 shares:** the trail still writes 87/13 in two places (BLOCKING DEPENDENCY, the paragraph after the first `---`).

## Your four questions

1. **New stale paragraphs / history relied on as live:** yes to both: F3 (a stale A1 sentence in an updated paragraph; live specification held in the "history"), F2, and the mixed 87/88 and 13/12 figures.
2. **Is "must not be quoted" enforceable?** No. Delete the history and keep a current-state section (F3).
3. **Does the Gate enforce the V1b exclusion?** No (F1, HIGH).
4. **A8 (no V1 checklist): deferring is the right call for the implementation checklist and the wrong call for two smaller things.** The checklist should be written against the mechanism that will be built,
   after Task 5 (b) fixes what the words are, and writing it twice is waste. But (a) the **Gate** must be written now, because it is what a reviewer will apply and it currently accepts V2, V1b and V1
   alike (F1); and (b) the heading and the checklist header should say plainly that the checklist below specifies **V2 only** and V1's is to follow, because at present a reader who takes the plan at its
   word ("Chosen first mechanism: two-phase") and follows the only checklist builds the variant the plan expects to reject. A two-line note above the checklist and a Gate paragraph cost nothing and remove
   the risk that the largest open item is also the least visible.

## What I did not check

The measured lane counts (`dad59f1b48`, 33.9, 64,635 against 64,857, the 36/42/22 split) and the join onto `task1-2-new-on-T`'s stamps: I checked the table's internal arithmetic (shares, 87.9%, +4.93, -15.67,
7.4%, 33.9 x c) and not the trace. `t3-topology` is sweeping the rest of the plan for stale figures; I read only Task 6. The 3-slot demonstration and the `kBusySeq` windows remain uncommitted (the plan now says so).
