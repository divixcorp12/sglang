# Stage B report — captured side-stream expert pull (synthetic plans)

## Status: DONE

Commits: `b37d33f1d8` (initial), `e633953b51` (fix round 1), both on
`codex/nvfp4-expert-stream-main`.

Files added, none of the excluded files touched:
- `python/sglang/srt/layers/moe/expert_gpu_pull.py`
- `test/registered/unit/layers/moe/test_expert_gpu_pull.py`
- `benchmark/kernels/moe/benchmark_expert_request_latency.py`

## 1. The join-load-bearing negative test — the headline finding

What was removed: the `pipeline.join_target(target)` call inside the captured
body, conditioned on a `join: bool` parameter passed to
`_capture_and_replay_delayed_fork`.

What actually happens when it is removed is **stronger than a bad byte
compare**: CUDA's own stream-capture machinery refuses to end capture at all.
`with torch.cuda.graph(graph): body()` raises at `__exit__` →
`capture_end()`:

```
torch.AcceleratorError: CUDA error: capturing stream has unjoined work
Search for `cudaErrorStreamCaptureUnjoined' ...
```

`test_removing_the_join_makes_graph_capture_itself_refuse_to_end` asserts
exactly this with `pytest.raises(RuntimeError, match="unjoined")`. This is a
capture-time hard error, not a runtime race: a graph containing an unjoined
fork cannot be built in the first place, so this failure mode can never reach
a replay in production. Verified first with a naive version of the test that
expected a byte-mismatch `AssertionError` instead — it failed with this
`AcceleratorError` before the assertion was ever reached, which is what drove
the rewrite.

Because that capture-time error is CUDA-specific machinery rather than
something guaranteed for every misuse, I added a second, independent
demonstration of the same load-bearing property in eager (uncaptured) mode,
where CUDA does not enforce anything: `_post_with_delay` forks a one-row pull
whose side branch spins for `torch.cuda._sleep(400_000_000)` cycles before
copying, then the origin stream is confirmed complete via its own event
(`origin_done.synchronize()`, never a device-wide sync) and the destination
is read immediately. Without a join, the observed bytes are the untouched
255 sentinel, not the source row —
`test_forking_without_a_join_leaves_the_destination_unwritten_when_observed_early`
asserts both that the observed bytes equal the sentinel and that comparing
them against the true source row raises `AssertionError`. This uses a
device-clock-bound delay, not a host `time.sleep`, so the outcome is
deterministic rather than a race against wall-clock scheduling.

`test_join_makes_a_delayed_fork_deterministically_correct` is the same
delayed-fork harness with `join=True`: it captures and replays successfully
and the destination is byte-exact against the source despite the artificial
400M-cycle delay, showing the join correctly waits out the delay rather than
merely happening to finish first.

## 2. Physical overlap — measured, not inferred

`test_physical_overlap_of_side_pull_and_origin_compute` records CUDA events
(`enable_timing=True`) around the actual copy kernel on the side stream and
around a 4096×4096 matmul on the origin stream, relative to a shared `t0`,
for two schedules: **overlapped** (compute starts immediately after forking
the pull, joins only at the end) and **serialized** (origin joins before
starting compute). Measured on this run:

```
overlapped:  copy=[0.0334, 0.2150] ms   compute=[0.1506, 2.4971] ms
serialized:  copy=[0.0277, 0.1523] ms   compute=[0.1544, 2.0586] ms
physical_overlap_ms = 0.0645   (min(copy_end,compute_end) - max(copy_start,compute_start))
serialized_gap_ms   = 0.0020   (compute_start - copy_end)
```

This is real, positive kernel-interval overlap from device timestamps: in the
overlapped schedule the matmul kernel started at 0.1506 ms while the copy
kernel was still running (it ended at 0.2150 ms), a ~65 µs window where both
were concurrently in flight on the device. The serialized schedule's own gap
is ~2 µs (essentially back-to-back, as expected when the join forces
ordering), confirming the two schedules are actually different at the
hardware level and the comparison is meaningful, not an artifact of stream
naming.

Stating the limits plainly: the overlap window is modest (65 µs) relative to
the matmul's own duration (~2.3 ms), mostly because the one-row 1 MiB copy is
short next to the chosen compute workload, so much of the compute proceeds
after the copy has already finished — that's still genuine concurrency, not
a larger number I'm rounding up. I did not attempt to tune workload sizes to
maximize the reported overlap; the numbers above are the first and only
overlap run in this stage, not a best-of-N.

### 2a. Is the 65 µs bound a design property, or a test-harness artefact?

Pressed on this directly: **it is a host-dispatch/issue-order artefact of the
eager measurement harness, not a GPU dependency, and not a property of the
fork/join design.** In the overlapped schedule, nothing on the origin stream
waits for `target.done` before `compute_start` is recorded — the origin
stream is free to run the matmul the instant its own prior node (the
`target.ready` record) is done. The 117 µs between `copy_start` (0.0334 ms)
and `compute_start` (0.1506 ms) is explained by CPU time: between those two
timestamps, the host has to sequentially issue `side_stream.wait_event`,
`copy_start.record`, the copy kernel launch, `copy_end.record`, and
`target.done.record`, all via separate Python/CUDA-API calls, *before* it
returns to the origin stream and issues `compute_start.record` and the matmul
launch. Each of those five calls costs real host dispatch time (Python +
`cudaLaunchKernel`/`cudaEventRecord` overhead), and that dispatch time is what
shows up as the gap — not any wait the origin stream is actually performing.
The serialized schedule's own back-to-back gap (0.0020 ms) confirms these
five-call sequences individually cost only a few µs each when nothing else is
interleaved; it's specifically the *interleaving* — issuing all of the side
branch's launches before the origin's next launch — that stacks up to 117 µs
here.

The structurally relevant fact is therefore: copy duration (≈0.182 ms) fits
roughly 13× inside compute duration (≈2.347 ms), and nothing observed here
rules out hiding the whole copy behind compute — the ceiling this run
measured is an artefact of eager launch order, not evidence of the design's
own limit.

That said, this eager harness is *not* the production path, and I have not
measured the number that would actually settle it. Production replay
launches the entire captured graph in one `graph.replay()` API call; the
GPU's own graph-execution engine schedules independent nodes against each
other without paying repeated host dispatch latency between them the way
five sequential eager Python/CUDA-API calls do. The 65 µs figure is
therefore likely a *pessimistic* bound on production overlap, not a
realistic one in either direction I can currently claim with confidence — it
could show much fuller hiding under replay, or it could reveal some other
graph-scheduling constraint I have not looked for. Measuring overlap under
actual replay (capturing the same timing-enabled events *into* the graph
alongside the production timing-disabled ones, then reading elapsed time
after several replays) is buildable but unwritten; I did not build it in
this stage. **Do not use 0.0645 ms as the design's overlap ceiling for any
Stage C claim** — treat it as a lower bound produced by an eager test
harness, with the real production number still unmeasured.

## 3. Test count

Post fix-round-1 (commit `e633953b51`):

```
23 passed, 15 warnings in 9.28s
```

Breakdown:
- `test_expert_gpu_pull.py`: 10 passed
  (fork/join byte-exact, join-capture-refusal, join-under-delay-correct,
  unjoined-eager-observed-sentinel, count-0/1×120-replay sweep,
  two-separate-graph-states, tag lookup/duplicate registration, `join_all`,
  capacity!=1 rejection, physical overlap)
- `test_expert_cache_transfer.py`: 13 passed — matches the stated baseline
  (13 passed in 14.75 s; this combined run took 9.28 s total for both files,
  consistent with shared process warmup).

(Original submission, commit `b37d33f1d8`, was 9+13=22 passed; §5 below adds
the `join_all` test and one more assertion, netting +1.)

## 4. Skips

None. `grep -i skip` over the full log returns nothing. No `pytest.mark.skip`,
no `pytest.skip()`, no collection errors. CUDA was available and used
throughout (RTX 5090, confirmed via the physical-overlap matmul and the
1 MiB-row copy timings above).

## GPU discipline

- `nvidia-smi --query-compute-apps` was empty (no PIDs) immediately before and
  immediately after the run; no contamination signal.
- One earlier run failed with `1 failed, 20 passed` due to a real bug I found
  and fixed in my own new code: `ExpertGpuPullPipeline.__init__` stored
  whatever `torch.device` it was given verbatim, so `ExpertGpuPullPipeline(
  torch.device("cuda"))` (no index) never equaled `segments.table.device`
  (always indexed, e.g. `cuda:0`), tripping the `__post_init__` device check
  for every caller that didn't pass an explicit index — including three of my
  own tests. Fixed by resolving to `torch.cuda.current_device()` when the
  index is `None`, matching the existing pattern in
  `benchmark_expert_cache_transfer.py`. Re-ran after the fix; still hit the
  capture-time `AcceleratorError` above, which led to the test rewrite in §1.
  Third run (this report's numbers) is clean: 22 passed, 0 failed, 0 skipped.

  **Caution for Stage C callers, not just a fixed bug:** the fix lives in the
  pipeline constructor, so any caller constructing `ExpertGpuPullPipeline`
  with a bare `"cuda"` string or an unindexed `torch.device("cuda")` is now
  safe. But anything that independently compares a device it was handed
  against a tensor's `.device` (as `__post_init__` does, and as other code in
  this area may) is exposed to the same trap unless it resolves the index
  first — a bare `cuda` device and its own `cuda:0` tensors will silently
  fail an equality check that looks like it should obviously pass. Worth a
  grep for other raw `torch.device` comparisons before Stage C wires this
  into serving code that may be handed device objects from multiple call
  sites.

## 5. Fix round 1 — regression coverage on where `post_target` records `done`

Reviewer finding (MEDIUM, not a live bug): three of the original tests
(`test_fork_join_byte_exact_with_independent_origin_compute`,
`test_count_zero_and_one_across_many_replays...`,
`test_two_separately_allocated_graph_states...`) call
`torch.cuda.synchronize(device)` before reading the destination, which waits
for *everything on every stream* regardless of the graph's own dependency
edges — so a regression that moved `target.done.record(...)` to before the
copy would be invisible to them by construction. The two tests with the
narrow, correct read pattern (`test_join_makes_a_delayed_fork_deterministically_correct`,
`test_forking_without_a_join_leaves_the_destination_unwritten_when_observed_early`)
called a hand-duplicated `_post_with_delay` instead of the real
`post_target`, so they weren't actually watching the real method either.

**Fix:** removed `_post_with_delay`. Both tests now call the real
`pipeline.post_target(target)`; the device-clock delay is injected via
`monkeypatch.setattr` on the module-level `copy_expert_row_segments_gpu`
reference that `post_target` calls internally, wrapping it with
`torch.cuda._sleep(_DELAY_CYCLES)` before delegating to the original. This
keeps the delay device-clock-bound (per your note on the doorbell
`@functools.cache` warm/cold hazard: the monkeypatch changes *what* runs at
the call site, not *how* the delay is measured — still no host-side
wall-clock race).

**Drove it red before calling it done**, as instructed. Temporarily moved
`target.done.record(self.side_stream)` in the real `post_target` to
*before* `copy_expert_row_segments_gpu(...)` instead of after. Reran
`test_expert_gpu_pull.py` alone:

```
5 failed, 5 passed, 15 warnings in 10.09s
FAILED test_fork_join_byte_exact_with_independent_origin_compute
FAILED test_join_makes_a_delayed_fork_deterministically_correct
FAILED test_count_zero_and_one_across_many_replays_and_changing_expert_ids
FAILED test_two_separately_allocated_graph_states_do_not_interfere
FAILED test_join_all_joins_every_registered_target_before_capture_ends
```

The actual failure symptom was stronger, and broader, than what the fix was
built to catch: every one of those five failed at **`capture_end()`** with
the same `cudaErrorStreamCaptureUnjoined` seen in §1 — including the three
full-device-sync tests the reviewer flagged as blind. The reason: once
`done` is recorded before the copy, the copy kernel becomes *trailing work
on the side stream launched after the event that joins it back to origin* —
CUDA's own capture validator treats that trailing kernel as itself unjoined,
regardless of whether anything ever reads its result. So this specific
reordering is caught structurally, at capture time, for every captured test
in the file, not only the two the fix targeted.

**What this actually establishes, stated plainly rather than as a soft
caveat:** the red-drive did not demonstrate the new narrow-sync coverage
catching anything. It showed that `capture_end`'s own connectivity validator
subsumes this entire mutation class for the *current* code shape. `post_target`
is one `wait_event` in, one kernel, one `record`, one `wait_event` out —
there is no third position for the `done` record: it either correctly
precedes the copy, or it leaves the copy as trailing unjoined work, which
CUDA rejects structurally before any test's assertions run. Recording `done`
on the wrong stream, or `join_target` waiting on the wrong event, collapses
to the same rejection for the same reason. And a wrong-data bug (the
computed bytes are wrong, not their timing) is orthogonal to sync-blindness
in the first place — a device sync changes *when* something is observed to
finish, never *what* was computed, so it was never the mechanism that could
mask that class either.

**So: the added narrow-sync coverage is not validated by mutation, and is
not currently validatable by mutation against this code shape.** That is the
state of the evidence, not a hedge. I looked for a discriminating mutation
(moving the destination read before `join_target` in the captured body) and
it does not work either: capture still succeeds (everything is joined
before capture ends, the read is merely ordered earlier in the body), but
the corruption is baked into whatever the origin read, so a full-device sync
afterwards reads the already-wrong buffer and a full-sync test catches it
too — refuted, not a counterexample.

Its value is real but forward-looking, not demonstrated today. Section 7.4's
target-disabled and model-tail cases, and Stage C's multi-op per-target
pulls, mean the side branch will eventually gain a second op. At that point
a `done`-placement bug can leave *some but not all* side-stream work joined
— structurally valid to `capture_end` (the graph is still fully connected)
but semantically wrong — which would not trip `capture_end()` and would be
caught only by a narrow-sync read against the real method. Provably inert
against today's one-kernel shape; provably relevant the day the side branch
grows a second op.

Reverted the mutation, confirmed byte-identical to the pre-mutation file
(`git diff` empty), reran: clean 23 passed, 0 failed, 0 skipped.

`join_all()` (LOW item): added
`test_join_all_joins_every_registered_target_before_capture_ends` — two
targets on one pipeline, captured body calls only `join_all` (never a
per-target `join_target`), verifies capture succeeds and both destinations
are byte-exact after replay.

`test_removing_the_join_makes_graph_capture_itself_refuse_to_end`'s missing
`del graph` (LOW item, no change made): confirmed — the exception fires
inside `with torch.cuda.graph(graph): body()`, so `_capture_and_replay_delayed_fork`
never reaches its `del graph` line for the `join=False` path, and no
explicit capture-abort recovery is attempted. All runs in this stage
(23/23, plus the mutated 5/10 run and its revert) show no order-dependence
symptom, but that's empirical, not structural: under a randomizing plugin or
a different pytest version this is unverified. Flagging per your ask rather
than changing it.

## Concerns

- The physical-overlap window (65 µs) is real but small at this row size;
  anyone using this measurement to justify Stage C net-gain claims should
  re-measure at the actual production row/tensor-count geometry (48 layers,
  NVFP4 row widths per §7.1 of the plan) rather than reusing this
  synthetic-benchmark number.
- I did not modify `expert_stream.py`, `expert_hot_cache.py`,
  `model_runner.py`, `environ.py`, or anything under `expert_prediction/**`,
  per the brief's do-not-touch list.
- Stage A's files (`expert_route_plan.py`, `expert_stream.py`,
  `expert_route_plan.cuh`, `test_expert_route_plan_fused.py`) were dirty
  throughout from a concurrent session; I did not read their diffs and
  nothing in Stage B imports them.
