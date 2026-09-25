# Doorbell copier: what you would be deciding

Written while the decision was open, and kept as written: it is the brief the merge was approved
from. Recorded here for the record rather than edited into hindsight.

**Status now.** Approved and merged, **disabled by default** (`SGLANG_MOE_EXPERT_DOORBELL`
defaults to false). `cc/doorbell-serving` landed on `master` as merge
commit `da3e9297be`, over three commits — `1a06114e87` (bounded drain and host fail-stop),
`0f20010cfe` (always-synchronize fail-stop check and the copier prime), `245c5e0d67` (tests).
Not pushed. The port 7867 server the user tested these changes on was stopped after the
measurements below were taken; the GPU went to another session.

The flag-off path was verified rather than assumed, because with the doorbell disabled it is the
only path that runs: `test_flag_off_gather_matches_the_pre_doorbell_path` and
`test_gathers_match_the_pre_doorbell_path_across_residency_updates` both pass, and the
pre-doorbell reference they compare against was diffed against `7de955329a` itself.

## What it buys

- **About 10% more transfer rate.** Measured on divix01 at production granularity (7 rows of
  2.76 MB, pinned host memory, 1500 iterations): copy engine **13.313 GB/s** versus the existing
  in-kernel path **12.081 GB/s**, a ratio of **1.102**. Both arms verified the bytes they moved.
- **Plus an unquantified overlap benefit.** The in-kernel path burns SMs for ~1.6 ms per gather;
  the doorbell frees them. That may matter more than the 10%, but it has not been measured.
- The earlier framing that the doorbell "unlocks most of the bus" is **refuted**. Both paths sit at
  the practical ceiling of this machine's PCIe gen3 x16 link, which trains to gen3 under load
  (37 samples, all gen3 x16).

## What it costs, and the part to weigh hardest

- **On the serving path the drain has never been observed to recover a request.** If the copier
  thread is slow enough that a resolve times out, the drain runs its entire budget and ends in
  fail-stop disable. Recovery works in the copier unit tests (a late copy lands mid-drain in 6
  microseconds); on the graph-gather path the copy did not execute until the drain retired, at
  every budget tested. One caveat, unresolved: the behaviour reproduces deterministically in the
  direct harness (8 of 8) but did **not** reproduce once under pytest, where the drain succeeded.
  That single run is unexplained, so this is 13 of 14 observations rather than a rule.
- **The failure mode is a process abort, not a degraded request.** When that happens the copier is
  disabled for the life of the process and the server deliberately aborts itself about 30 seconds
  later. The server can vanish mid-session. It is safe by design, but it is not graceful.
- **A late copy can land after resolve has returned.** The protection is the abort, not the
  absence of the write: once the thread has committed a request it issues the copy, and a drain
  that gives up afterwards cannot unmake it.
- **The hold is unexplained, not fixed.** Five candidate mechanisms were tested and all refuted:
  cold kernel launch, copy-engine starvation by a resident kernel, CUDA-graph replay, unpinned or
  wrong-kind memory, and hardware channel multiplexing (`CUDA_DEVICE_MAX_CONNECTIONS=32` changed
  nothing). The behaviour is bounded and fail-safe; the cause is unknown.

## A named risk: two safety rules that are true today and silent when broken

The layer has two rules that callers must honour, which nothing in the code enforces or detects.

- **One outstanding post per tag.** If a caller posts again on a tag whose earlier request is still
  outstanding, the earlier request is **silently orphaned**: the per-tag sequence word is simply
  overwritten, resolve then matches only the later request, and the earlier one is never reported
  to anyone. **No counter fires and no error is raised.**
- **The reservation rule.** A slot named by an outstanding plan must not be read before its resolve
  returns nor written by anyone else until then, and plan tensors must not change between a post
  and its resolve. Also unchecked at runtime.

Both are honoured today, because the only caller is our own gather path. The risk is a future
caller that does not honour them, and nothing will tell them — the failure is silent rather than
loud.

An option, if you want the silent case made visible: `post()` could compare against the tag's
outstanding sequence and increment a counter instead of overwriting without comment. That makes a
violation detectable in the counters rather than invisible. It has not been built, and this brief
makes no recommendation either way — the question for you is whether a silent failure mode is
acceptable in something you may merge.

## Status of the evidence

- The live server showed the doorbell servicing traffic normally. The figures above were taken
  early in that run; over its full life it reached **340,657 posted, 340,560 serviced, 853,086
  rows copied**, with zero timeouts, zero drains, zero drain timeouts, zero copy errors, zero
  late completions and no abort. The remaining 97 are 96 `skipped_abandoned` (capture-time
  resolves, expected) and 1 `skipped_overrun`.
- Measured against a known token count after the user finished testing (two controlled 2,000-token
  generations, temperature 0, count taken from the response): **47.976 gathers and 479.76
  requested rows per token**, both prompt-independent; hit rate **0.86 and 0.79**, bytes per token
  **171.6 and 259.9 MiB**. Hot cache 12 GiB, 4,180 slots. Measured PCIe rx over those windows was
  **5.938 and 7.684 GB/s** against a derived **5.062 and 6.226 GB/s** — a ~20% disagreement that
  is reported, not reconciled. True PCIe duty cycle is **NOT MEASURED**: 1 Hz sampling cannot
  resolve it, and the ratio against the 12.081 GB/s ceiling is a mean-throughput ratio, not a
  duty cycle.
- One test discrepancy is **open**: the gather-path test failed once with the drain succeeding,
  while 8 of 8 direct repeats showed it failing as expected. Until that is explained, the claim
  "the gather path never recovers" is supported by 13 of 14 observations, not by all of them.
- Not measured: bytes per token against a known token count, PCIe duty cycle during decode, and
  whether the decode loop is transfer-bound at all. The 10% figure is a microbenchmark of the
  transfer pattern, not a decode-loop result.
- Not measured: whether the copy engine is actually carrying bytes in serving. The copier's own
  `bytes_copied` and `rows_copied` are surfaced in the metrics file, so a figure exists — but it
  is not an independent one. `bytes_copied` equals `decode_h2d_bytes` **to the byte** in every
  window measured, because `rows_copied` equals `miss_rows` and both figures are that same row
  count times the same 2,764,800-byte row constant. They agree **by construction and could not
  have disagreed**, so their agreement confirms nothing. `gather_copy_engine_bytes` structurally
  cannot see doorbell copies at all (it folds only the eager path), and `h2d_bytes` is derived
  arithmetic from miss counts. The only independent measurement of bytes on the wire is the
  external PCIe counter, which disagrees with the derived figure by ~20%. Nothing in the system
  measures the doorbell's transferred bytes directly.
