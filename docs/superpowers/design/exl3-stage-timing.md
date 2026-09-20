# Stage timing for the native RAM-miss service

Input to storage-cpu-pipeline Task 1 (handoff section 7 point 3). Written from a
working implementation; read section 6 first for what that implementation is and
how weak the evidence for it is.

## 0. Status of the code this describes

The implementation exists as commit `b4257dd412`, on top of the extents commit
`18949eafad`. It was written first against the pre-extents reader
(`d255ddf079`), then ported onto the per-extent reader when Task 2 of the
companion plan landed mid-work. The instruction to stop and write this note
arrived after the port was committed. Keeping, revising or dropping that commit
is the lead's call; every file:line below is in `exl3_ram_miss_host.cpp` at
`b4257dd412`. Before that commit the same points are the statements named in
each row, so the references survive a re-implementation.

## 1. Stages and where each is stamped

All times are `now_ns()` (`CLOCK_MONOTONIC`, host). `time.monotonic()` reads the
same clock, so the trace's `t` and the stages compare directly. No GPU timestamp
is compared with a host one anywhere. A stage the request never reached is 0.

| Stage | Point | Line | Per-extent note |
|---|---|---|---|
| observed | `begin_stage`, called straight after the head-word check passes in `pump_demand` / `pump_advice` | 997 (called 814, 838) | Unchanged. Before the seqlock read and the overrun handling. |
| reserved | in `serve()`, after the mutex block that reserves slots, before the read | 1141 | Unchanged. Includes any wait on `mutex_`. |
| submit | just before the first `submit_and_wait` of a batch | 321 | With extents this is the first SQE batch of all a row's parts. SQEs are prepared just before it (loop at 312), so prepare time is inside reserved-to-submit. |
| first_cqe | the wait that returned at least one completion, stamped right after it returns and before reaping | 334, stored 377 | Now the first completion of ANY extent of ANY row in the batch, not of a row. |
| last_cqe | the same stamp on the last iteration that reaped anything | 334, stored 377 | Now the last completion over all extents, so it is the slowest drive's tail. A resubmitted short extent moves it. |
| pack_start / pack_end | around the six-slab memcpy loop | 409 / 422 | Publication is still per batch, after every extent of the batch succeeded. |
| mapped | end of `serve()`, after slots are marked READY and the slot map is published | 1182 | Unchanged. |
| done | `end_stage`, right after `store_release(kDemandDone)` (or `kAdviseDone`) | 1012 (called 829, 865) | This is when the device wait can release. The stamp is taken after the store, so the store is not delayed. |

Definition that matters once reads are per extent: first and last CQE are taken
per BATCH of rows, over all extents, and mean "the wait returned", not "the
drive finished". A per-row completion time does not exist in the record. Per-row
readiness is a distinct event the overlapped pipeline (Task 4) will create; this
record does not model it and must be extended, not reinterpreted, when it does.

Multi-batch requests. A request spans several io_uring batches only past 8
rows, or one row per batch for an advisory. The stamps from `submit` through
`pack_start` are the FIRST batch's and `pack_end` is the LAST batch's, so
`observed <= reserved <= submit <= first_cqe <= last_cqe <= pack_start <=
pack_end <= mapped <= done` always holds. The three `*_ns` sums cover every
batch and are what a decomposition of `observed..done` must add up: never derive
phase durations from stamp differences on a multi-batch request.

No completion-generation field. The companion plan deliberately adds none, since
`RowReader::read` returns with the ring empty, so no completion outlives its
request. Generations arrive with the overlapped pipeline (Task 4). The record
carries none and nothing in the instrumentation assumes one.

## 2. Record shape, and why fixed-size

`StageRecord` (line 67): 34 int64 words, no pointers, no allocation.

    seq kind row ok rows batches backlog prev_done
    observed reserved submit first_cqe last_cqe pack_start pack_end mapped done
    submit_to_first_cqe_ns first_to_last_cqe_ns pack_ns bytes extents
    drive_dev[4] drive_bytes[4] drive_extents[4]

- `kind`: 0 demand, 1 advisory, 2 touch (an unarmed record: no read).
- `row`: the streamed row, not the layer id; Python maps it with `layer_ids`.
- `backlog`: records already posted behind this one when it was observed.
- `prev_done`: the `done` of the request served just before, 0 for the first.
- `bytes` is what the reads returned, summed over drives; `extents` counts reads
  issued (a zero-length part issues none and is not counted).

Fixed size and int64 only, for four reasons: the service thread writes it with
no allocation in the hot path; one `memcpy` puts it in a preallocated
single-producer ring (dropped and counted when full, capacity set at enable); the
same bytes are a row of a torch int64 tensor, so Python drains with one call and
no marshalling; and the layout is checkable (`STAGE_FIELDS` must match, and
`_stage_words()` fails on a mismatch). Adding a field means appending words and
bumping both sides, which the check forces.

Trace line, on the existing `SGLANG_DSV41_EXPERT_TRACE_PATH` file, written by
`Exl3StreamTrace.record_ram_miss_requests`: `kind: "ram_miss_request"`, with
`request`, `stages_ns`, `prev_done_ns`, `spans_ns`, `bytes`, `extents`, `drives`
and `t`. It has no `tokens`, because it is not a forward call, and
`scripts/dsv41/tier_sim.py::load_trace` skips it so `live_summary` and
`simulate` see the same G and f as before. Any other reader of that file that
groups by `forward` must do the same, or a stage line becomes `forward[0]`.

## 3. Per-drive attribution keys on the opened file's device

`RowReader::open` (line 240 onward) `fstat`s every fd it opens and keys a drive
slot on `st_dev`, in first-opened order (line 245, 249). Per extent, the reader
looks up `file_drive_[extent.file]` and adds that extent's `done` bytes and one
extent to that slot (line 390). Attribution is by the device of the file that
extent actually read.

It deliberately does NOT key on a root index or on `file % parts`. The extent
table numbers files shard-major, root-minor, but the reader never has to know
that, so the attribution stays correct if the numbering, the part count, or the
policy that assigns parts changes, and it is right by construction for a
checkpoint that is not mirrored at all (one slot).

Limits, stated so nobody reads more into it:
- `st_dev` names a filesystem, not a physical disk. A RAID or LVM span is one
  drive. Two mirror roots on one filesystem are one drive; that is correct for
  what the kernel sees but not what the operator meant by two roots.
- More than 4 distinct devices fold into the last slot with `dev = -1`.
- A failed `fstat` gives `dev = -1` and shares a slot with any other such file.

## 4. What could not be cleanly separated

- Posted versus observed. The post is a GPU store into pinned memory and has no
  host clock. The only fix is to compare a GPU timestamp with a host one, which
  is not allowed. The record has `observed`, `backlog` and `prev_done`. If the
  service was idle, observed is within one poll period of the post (about 50 us
  asleep, a pause-loop iteration when spinning). If it was busy, `observed -
  prev_done` near zero with `backlog > 0` is queueing. Queueing before observed is
  therefore separable from service time only as a bound.
- Queueing versus submit inside the service. `reserved` to `submit` mixes waiting
  for `mutex_`, the `abandon()` check, SQE preparation and the delay injection
  used by tests. They are not distinguishable at the current structure. Reserved
  to submit was about 1.5 us on tmpfs; on NVMe it is expected to be small, but
  that is unmeasured.
- First and last CQE are when the wait returned, not when the device completed.
  Wakeup latency is inside submit-to-first-CQE.
- Per-row completion, for the reason in section 1.
- Skipped advisories get no record (they are never served); the existing
  `advisories_skipped` counter is their only trace.

## 5. What was judged too costly

Nothing. With the trace off, the service does one relaxed atomic load per
request in `begin_stage` and a null-pointer check per batch in the reader: no
clock read, no store. With it on there are about 12 clock reads per request
(vDSO, about 20 ns each), a 272-byte record copy and one ring push. On tmpfs,
serving 4000 requests of 3 rows each, five alternating runs pinned to cores
2-3, median per-request time was 9.2-9.6 us off and 9.5-9.7 us on, about +3%,
against reads that cost milliseconds on NVMe. The trace-off build was NOT
compared with a build without the instrumentation, so "costs nothing when off"
is an argument from the code, not a measurement.

Deliberately not added: a stamp per CQE (a clock read inside the completion
loop) and per-row timestamps inside a batch. Neither is needed for the stage
split and both sit on the loop the resubmit and drain invariants depend on.

The instrumentation only reads the clock and adds to the record. It does not
change what is submitted, reaped, drained or copied, so ring-empty-on-return,
short-read resubmit, soft-error retry and drain-on-error are unaffected by
construction. That too is argued from the diff and the pre-existing tests, not
proven independently.

## 6. What was verified, bluntly

Built and run: the C++ was JIT-compiled locally (g++, liburing) and the CPU
tests ran. CPU only; no GPU, no ssh, no real NVMe. Nothing here says anything
about production timings, the queueing split, or O_DIRECT behaviour on the real
drives (the direct-I/O test uses whatever filesystem `tmp_path` is, and tmpfs
skips it).

The environment is NOT a normal one, and the evidence is weaker for it:
- The machine's Python had no `tvm_ffi`, `transformers`, `dill`, `loguru`,
  `sentencepiece`, `compressed_tensors`, `flashinfer-python`, `nvidia-ml-py`,
  `gguf`, `openai`, `xgrammar`, `einops` and several smaller packages. I
  installed them with `pip install --target` into a scratch directory and put
  it on `PYTHONPATH`. `apache-tvm-ffi` was pinned to 0.1.11 to match
  `pyproject.toml`; the others were unpinned latest.
- `sgl_kernel`'s compiled ops do not load here (undefined-symbol ABI mismatch
  against the installed torch). I replaced the package with a stub module whose
  every attribute is a permissive object that raises only if called. The exl3
  CPU tests never call it, but the stub means import-time behaviour of that
  package was not exercised.
- Tests ran from an `rsync` copy of the worktree in a scratch directory, not
  in-tree.
- torch is 2.14.0+cu130 from the system Python, not the repo's pinned version.

Result: 344 tests passed, none failed, over the exl3 kernel test files, the exl3
and ram_miss tests under `layers/moe`, and `test_exl3_moe_stream_mode.py`,
including every pre-existing fault, thread and mirror test. Before the C++
change, the 16 new tests failed and the 24 existing ones passed. I did not run
the whole `layers/moe` directory, so the known 11 pyarrow collection errors
never came up and are not evidence either way. One of my own tests
(`test_exl3_ram_miss_device_args`) first failed because it parses every
`constexpr` in the file as a page constant; the record's word count is
therefore a function, not a `constexpr`.

Not run: the manual and GPU tests (`test/manual/dsv41`), the device kernels with
the trace on, an in-tree run in a properly provisioned environment, and Task 1
Step 6 (a baseline on divix01). The plan's number, the split of the RAM-miss
wait, does not exist yet.

The reference to the extents commit: the ops wrapper and the test fixtures had
to be read at `18949eafad`; the reader-level test helper `_bytes_read` sums over
`tables.extents` and skips zero-length parts (an empty part's offset can lie
past EOF and would otherwise subtract).
