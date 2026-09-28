# Copy-mechanism sweep: divix01 results (RTX 5090, sm_120f, CUDA 13.4 toolkit / torch cu13.0, PCIe Gen3 x16)

Full run 2026-09-27T18:37 at commit 6c75e547b2, in the private worktree `wt-xfer`. `sglang.__file__` was
`/data/models/slang/nvfp4-work/wt-xfer/python/sglang/__init__.py`. The link was Gen3 x16 at start. 30 timed reps per
cell, 167 cells. `mech_bench.py` exited 0, and so did `mech_report.py`: no cell above the ceiling, no unsafe method,
fresh check not blind.

## 1. `mech_report.py gen3/divix01.jsonl` (verbatim)

```
# divix01: theoretical 15.75 GB/s, measured ceiling 13.686 GB/s
serial acquire 780.096 ns, flag RTT p50 1440 ns, min 1408 ns
BDP (ceiling x RTT min, primary) 19270 B; BDP (ceiling x serial acquire) 10676 B

| method | best GB/s | knee (bytes in flight) | share of measured |
|---|---:|---:|---:|
| ce_batch | 13.686 | None | 1.0 |
| ce_batch_small | 2.684 | None | 0.196 |
| ce_each | 13.622 | None | 0.995 |
| ce_each_small | 3.979 | None | 0.291 |
| cw_real | 3.808 | 4096 | 0.278 |
| ldgsts | 12.308 | 32768 | 0.899 |
| sm_cv16 | 12.309 | 16384 | 0.899 |
| sm_cv32 | 12.309 | 32768 | 0.899 |
| sm_small | 6.69 | 32768 | 0.489 |
| sm_weak16 | 12.31 | 16384 | 0.899 |
| sm_weak32 | 12.309 | 32768 | 0.899 |
| tma | 12.302 | 32768 | 0.899 |

named cells: {'cw_pattern': 11.599, 's_pattern': 12.294}

best GB/s by bytes in flight (K = the method's knee)

| method | 4 KiB | 8 KiB | 16 KiB | 32 KiB | 64 KiB | 128 KiB | 256 KiB | 512 KiB | 1024 KiB | 2048 KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cw_real | 3.808 K |  |  |  |  |  |  |  |  |  |
| ldgsts |  | 5.679 | 10.311 | 12.291 K | 12.308 | 12.308 | 12.308 | 12.308 |  |  |
| sm_cv16 | 4.956 | 8.691 | 11.832 K | 12.298 | 12.309 | 12.307 | 12.309 | 12.308 | 12.305 | 12.304 |
| sm_cv32 |  | 8.301 | 11.648 | 12.303 K | 12.308 | 12.309 | 12.308 | 12.307 | 12.306 | 12.305 |
| sm_small |  |  |  | 6.69 K |  |  |  |  |  |  |
| sm_weak16 | 4.497 | 8.368 | 11.714 K | 12.3 | 12.305 | 12.31 | 12.308 | 12.308 | 12.304 | 12.303 |
| sm_weak32 |  | 8.258 | 11.652 | 12.303 K | 12.309 | 12.308 | 12.307 | 12.308 | 12.308 | 12.303 |
| tma |  | 5.873 | 10.474 | 12.279 K | 12.298 | 12.302 | 12.302 | 12.298 | 12.282 |  |
```

The fresh check passed as required. The `.nc` control read stale bytes (`fresh: false`). Every swept method read the
bytes the host rewrote mid-kernel: `sm_cv16`, `sm_cv32`, `sm_weak16`, `sm_weak32`, `ldgsts` and `tma`. The SASS
assertion also held: the control kernel has no `CCTL`, and all 24 swept copy kernels have one. **What the fresh check can and cannot see.** The only negative control is `.nc`, so the check is
shown to detect a stale L1 line and nothing else. `cp.async.cg` (ldgsts) and TMA bypass L1, so their `fresh: true`
would come out the same without the acquire or without `fence.proxy.async.global`. A missing proxy fence has no
control here. Their safety rests on code review of the fence and acquire placement, not on this check. Read latency: a serial
acquire of a pinned host word takes 780.1 ns; a device word takes 108.9 ns. Flag round trip: p50 1440 ns, min 1408 ns
(240 rounds).

## 2. Commands

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && rm -f analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/mnt/nvme1/tmp-ests/xfer-tmp flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python analysis/dsv41-drive/copy-mechanism/mech_bench.py --repo $PWD \
     --probe analysis/dsv41-drive/copy-mechanism/gen3/probe.json --out analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl \
     2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}" \
  && python3 analysis/dsv41-drive/copy-mechanism/mech_report.py analysis/dsv41-drive/copy-mechanism/gen3/divix01.jsonl; echo REPORT_EXIT=$?'
# probe (gen3/probe.json), run at 94f3ff5863 in wt-xfer-a under the same lock, TMPDIR and PYTHONPATH:
python analysis/dsv41-drive/copy-mechanism/probe.py --repo $PWD --out analysis/dsv41-drive/copy-mechanism/gen3/probe.json
```

`TMPDIR` is on NVMe because divix01's root volume was full on 2026-09-27. A first probe run without it failed every
build (nvcc could not write `/tmp`); that run was killed and its output discarded.

## 3. Probe (`gen3/probe.json`)

A probe is ok when it built, ran, and copied the right bytes.

| probe | what | ok | SASS load opcodes |
|---|---|---|---|
| `v8_weak` | 32-byte weak `ld.global.v8.u32` of host memory | yes | `LDG.E.ENL2.256` |
| `v8_cv` | 32-byte `ld.global.cv.v8.u32` of host memory | yes | `LDG.E.ENL2.256.STRONG.SYS` |
| `ldgsts` | `cp.async.cg` host -> shared | yes | `LDGSTS.E.BYPASS.128`, `LDGDEPBAR` |
| `bulk` | `cp.async.bulk` (TMA) host -> shared, after `fence.proxy.async.global` | yes | `UBLKCP.S.G` |
| `batch` | `cudaMemcpyBatchAsync`, 4 x 1 KiB H2D | yes | (host API) |

Every mechanism exists on sm_120 and works from pinned host memory. `batch` must be issued on a non-default stream:
on the legacy NULL stream it returns `invalid argument`, as the CUDA 13.4 header documents.

Two harness defects were found and fixed before the run:

- **The `.nc` control was blind.** The fresh check's midpoint polled with `ld.acquire.sys`, which on sm_120 compiles
  to `LDG.E.STRONG.SYS` + `CCTL.IVALL`. The CCTL invalidates the whole SM's L1, so the control's `.nc` second pass read
  fresh bytes. The harness's own acquire was hiding the staleness it was meant to show. The control now polls with
  `ld.relaxed.sys` (no CCTL), and the sweep refuses to run unless the SASS shows that split (`README.md`).
- **The batch probe ran on torch's default stream.** Its first run reported `invalid argument`, which is the
  legacy-NULL-stream refusal, not a missing mechanism.

## 4. Part A rows of the Decision table (divix01, Gen3)

Every row is decided per host. A "no" on Gen3 says nothing about Gen5. The rows that also need Part C say so. Part C
is gated on the expert-stream-native-sync merge and has not run.

- **Deeper unroll in S (U=1 -> 4).** Condition: `sm_cv16` G8 U4 >= 1.05 x `s_pattern`. Measured 12.306 vs 12.294
  (1.001x < 1.05) -> **no (on Gen3)**, whatever Part C's copy share turns out to be. G8 U1..U16 spans only
  12.294-12.308, and `sm_weak16` G8 matches it.
- **cp.async ring or TMA in S.** Condition: best `ldgsts`/`tma` cell with <= 32 KiB per block in flight >= 1.10 x the
  best `sm_cv*` cell. Measured: best `ldgsts` G8 stages 2 = 12.308, against best `sm_cv*` 12.309 (0.9999x < 1.10)
  -> **no (on Gen3)**. Both passed the fresh check. No bulk path beat 12.5 GB/s: TMA's best cell is 12.302. Every SM
  path stops at about 12.31 GB/s, 0.90 of the copy engine's 13.69.
- **Wider copy-wait SM reads.** `cw_pattern` is contiguous and 4-deep, which is **not CW's real shape**. It ran at
  11.599 GB/s, below 0.90 x the measured ceiling (12.317), so the A half of the condition is met, but that decides
  nothing on Gen3. Every SM cell, including G32 U16 with 2 MiB in flight, is below 0.90 x the measured ceiling,
  because the measured ceiling is the copy engine's 13.686 while every SM path plateaus at 12.31 (0.899). Against the
  SM plateau, `cw_pattern` is 11.599 / 12.309 = 0.942, which does not meet the condition. The threshold should compare
  against the SM plateau (a note for the plan owner). What the cell does show is
  that a single block with 16 KiB in flight is bound by latency, not by the link. 16384 B / 1408 ns (the minimum
  round trip) = 11.64 GB/s, the same as the measured 11.6. Every 16 KiB cell lands in 10.3-11.8 GB/s whatever its
  grid or width. The streaming kernel (S, 32 KiB in flight) is bound by the link on Gen3 instead: 32 KiB cells run at
  12.27-12.30 GB/s.

  CW's real shape is the `cw_real` cell: the production `copy_wait_read` over CW's four small tensors per lane. That
  is 20,480 B through one 4-deep 16 KiB pass, then 9,216, 10,240 and 4,608 B through the 16-byte fallback, at about
  4 KiB in flight. Per lane: 1 lane 22.1 us (2.02 GB/s), 2 lanes 16.3 us (2.73), 4 lanes 13.3 us (3.36), 8 lanes
  11.7 us (3.81).

  Those reads are hidden. DSV41_REFERENCE.md §27.15 (node-mode trace `sm-small-B-node-20260926-034122`) records CW
  spin at 54.6 ms, spent waiting on CopyDone, and CW ending 4.96 us (p50 per layer) after the last copy lands. The C
  half (`cw_wait_ns` > 0 on >= 10% of copy-engine requests) needs Part C. -> **Not decided on the measured
  conditions (C pending). No CW change is recommended.**
- **`cudaMemcpyBatchAsync` in the copy thread.** Condition: `ce_batch_small` >= 1.5 x `ce_each_small` GB/s, or host ns
  per call <= 0.7 x, at 4 lanes. Measured 2.354 vs 3.218 GB/s (0.73x) and host 57,083 vs 40,660 ns (1.40x) -> **no
  (on Gen3)**, whatever Part C finds. Caveat: the small-segment copy-engine GB/s mostly measures host enqueue
  time, because `e0` is recorded before the API loop (4 lanes: host 40.7 us vs event 55 us for `ce_each_small`, 57.1
  vs 76 us for `ce_batch_small`). The two measures are therefore not independent, and the verdict rests on the
  host-time arm (1.40x). The batch call is slower at every lane count: 1 lane 1.33 vs
  1.45 GB/s; 8 lanes 2.68 vs 3.98 GB/s, host 101.7 vs 74.8 us. On the large segments batch matches per-segment
  copies on the device (13.69 vs 13.62 GB/s at 4 rows) but costs 1.70x the host time (85.6 vs 50.3 us).
- **Fewer, larger pieces / more, smaller pieces.** Condition: Part C only -> **pending (Part C gated)**.
- **PDL on the chain** (Part B, `../chain-pdl/results.md`). The first skeleton bound was 1.48 us/layer (mode 2,
  work_ns 2000), below the 2 us gate. The pre-wait probe changes this. With 200-500 ns of prologue per stage, the early
  trigger saves 3.26-5.25 us/layer, which crosses the gate. The implicit trigger saves 0.77. Both are at most 0.31% of
  the step, below the 1% ship bar. -> **gate met for the early trigger if real prologues are >= ~200 ns; Task 8 still
  gated on the merge.**

### Did the Gen3 expectation hold?

- **Every SM method at 12.1-12.4 GB/s:** held. `sm_cv16`, `sm_cv32`, `sm_weak16`, `sm_weak32`, `ldgsts` and `tma`
  all plateau at 12.30-12.31.
- **Copy engine at 13.5-13.8:** held. `ce_each` 13.54-13.62, `ce_batch` 13.58-13.69.
- **Knees near the bandwidth-delay product of latency x ceiling, about 8.7 KiB:** did not hold. The 16 B methods reach
  95% of their plateau at 16 KiB in flight; the 32 B, `ldgsts` and `tma` methods only at 32 KiB. The cells that break
  the expectation are every 8 KiB one, at 5.7-8.7 GB/s. The serial acquire (780 ns) understates the latency a loaded
  copy sees. `in_flight / GB/s` gives the effective time per window: 0.83 us at 4 KiB, 0.94 us at 8 KiB, 1.41 us at
  16 KiB (= the 1408 ns round trip), and 2.66 us at 32 KiB, where the link rather than latency is the limit. The
  primary BDP is therefore the ceiling x the minimum round trip: 19,270 B, not 10,676 B. The Gen5 argument at the top
  of the plan used the 711 ns figure, so re-derive its BDP from that host's measured round trip.
- **A bulk path above 12.5 GB/s:** none. TMA's best is 12.302 and `ldgsts`'s 12.308.

### Weak loads after the acquire vs `.cv`

`sm_weak*` (plain `ld.global` ordered after the acquire) is **contract-valid**. `.nc` is **not contract-valid,
informational only**.

Weak does not beat `.cv`. Best 12.310 vs 12.309 GB/s; the same knees (16 KiB at 16 B, 32 KiB at 32 B); slightly
lower at small in-flight sizes (4 KiB: 4.50 vs 4.96 GB/s).

### Small segments (the four small tensors, 44,544 B per lane)

The copy-engine columns below are bound by host enqueue time (see the batch row above).

| lanes | `ce_each_small` GB/s (host us) | `ce_batch_small` GB/s (host us) | `sm_small` GB/s (G8 U1) | `cw_real` GB/s (us per lane) |
|---:|---:|---:|---:|---:|
| 1 | 1.448 (16.3) | 1.329 (18.3) | 2.111 | 2.019 (22.1) |
| 2 | | | | 2.731 (16.3) |
| 4 | 3.218 (40.7) | 2.354 (57.1) | 5.364 | 3.360 (13.3) |
| 8 | 3.979 (74.8) | 2.684 (101.7) | 6.690 | 3.808 (11.7) |

## 5. Follow-up probes (2026-09-27, commit 3ee723bd8b)

Run in `wt-xfer` under the exclusive `cc-gpu.lock` with `TMPDIR` on NVMe. Command:

```bash
mech_bench.py --repo $PWD --probe gen3/probe.json --out gen3/divix01-probes.jsonl --only sm_line,tma   # MECH_EXIT=0
```

The report run (`mech_report.py gen3/divix01-probes.jsonl`) exited 0. In the same run: `nc_control` stale, `tma` fresh,
serial acquire 706.5 ns, RTT p50 1408 ns.

### 5a. Partial-line reads: why the SM path stops at 12.31 GB/s while the copy engine reaches 13.69

**Hypothesis tested.** The SM's sysmem path fetches whole 128 B lines, and the host answers each with 128 B
completions, while the copy engine gets completions at the full max payload.

**Probe.** `sm_line<T>` is the `sm_cv16` loop, `.cv` 16 B loads, at two plateau shapes: G8 U8 (256 KiB loaded in
flight) and G32 U4 (512 KiB). It loads only the first T bytes of every 128 B line, over the same 2.1 GB source and
213 MB of rotated destinations. Useful GB/s counts the bytes loaded; line GB/s counts lines touched x 128 B. The two
shapes agree to within 0.4%, so only G8 U8 is shown:

| bytes touched per 128 B line | time (ms, 416,112 lines) | useful GB/s | line GB/s | lines (requests) per second |
|---:|---:|---:|---:|---:|
| 16 | 1.760 | 3.784 | 30.27 | 236.5 M |
| 32 | 1.761 | 7.564 | 30.25 | 236.4 M |
| 64 | 2.634 | 10.110 | 20.22 | 158.0 M |
| 128 | 4.324 | 12.319 | 12.32 | 96.2 M |

**What the data shows.**

- **Neither stated prediction holds.**
  - The whole-line hypothesis is refuted. Line GB/s would stay about 12.3 if the SM fetched whole lines. Instead it
    reaches 30.3 GB/s at 16 B and 32 B, nearly twice what a 15.75 GB/s link can carry. So the SM does not fetch lines
    it only partly reads; it requests the touched sectors.
  - Useful GB/s does not stay near 12.3 at 32 B either: it is 7.56. So the SM is not sector-limited at the link rate
    either. `mech_report`'s verdict is "mixed" (32 B / 128 B = 0.61).
- **At 16 B and 32 B the time is the same to 0.05%** (1.760 vs 1.761 ms). The limit there is a fixed request rate,
  236 M requests per second, whatever the request size up to 32 B. That is 4.23 ns per request, or 1.08 us per request
  if 256 are outstanding. That matches a read's latency on this link: 707-780 ns serial, 1.34-1.41 us round trip.
  **This supports the 8-bit-tag hypothesis for small requests.** The GPU has `10BitTagReq` disabled (below), so it can
  keep at most 256 non-posted reads outstanding, and 256 / 1.08 us = 236 M/s.
- **At full lines the tags are not the limit.** Touching 128 B of a line needs 96 M requests/s, 41% of the rate the
  small-touch cells show the GPU can issue, and the plateau there is 12.32 GB/s. So the 12.31 GB/s SM ceiling comes
  from how efficiently each request is carried on the link, not from the number outstanding.
- **Completion size.** The 12.32 GB/s plateau lies between the predictions for 64 B completions (11.6 GB/s) and 128 B
  completions (13.2 GB/s): payload / (payload + 20 B TLP overhead) x 15.754 x 0.97 for link-control traffic. The copy
  engine's 13.69 needs 128-256 B completions (256 B predicts 14.2). The root port's 64 B RCB lets it split completions
  at 64 B boundaries. Consistent with "SM reads get smaller completions than copy-engine reads", but not conclusive:
  no TLP-level capture was taken.
- **At 64 B touched,** 158 M requests/s and 10.1 GB/s fit neither limit cleanly: under the tag rate, and below the
  64 B completion prediction. It is recorded, not explained.

**PCIe configuration.** `sudo -n lspci -vvv` was refused on divix01 ("a password is required"), and no other route was
tried. The values below are the user's own `lspci -vvv` readout:

| device | LnkCap | LnkSta | MaxPayload (DevCap / DevCtl) | MaxReadReq | RCB | 10-bit tags |
|---|---|---|---|---|---|---|
| GPU 37:00.0 (RTX 5090) | 32 GT/s x16 | 8 GT/s (downgraded) x16 | 256 / 256 B | 4096 B | 64 B | DevCap2 10BitTagReq+, DevCtl2 10BitTagReq- (disabled); ExtTag+ |
| Root port 36:00.0 (Intel Sky Lake-E Root Port A) | 8 GT/s max | | 256 B (DevCtl) | 128 B (only for reads the port itself originates) | 64 B | DevCap2 10BitTagComp- |

Root port cache line size: 64 B.

- **The Gen3 cap is the CPU's root port.** The GPU is Gen5-capable (32 GT/s) and trained down to 8 GT/s.
- **Completion-size arithmetic.** 64, 128 and 256 B completions predict about 11.6, 13.2 and 14.2 GB/s. The measured
  SM 12.31 and copy engine 13.69 fall inside that range: consistent, not conclusive.
- **10-bit tags.** The root port cannot complete 10-bit tags, so the GPU runs with 8-bit tags: 256 outstanding reads,
  16-32 KiB in flight at 64-128 B per request. The partial-line data supports this as the limit on small requests (the
  236 M/s rate). It does not make tags the limit at the 12.31 plateau, which needs well under that rate.
- **What to check on a Gen5 host:** 10BitTagReq+ on the GPU **and** 10BitTagComp+ on its root port (otherwise the
  256-read cap stays), the MaxPayload of both ends, and LnkSta. See `README.md`, "New host pre-flight".

### 5b. TMA without the proxy fence (`tma_nofence`)

**Probe.** `tma_nofence` is the `tma` fresh check's exact kernel, stages 4, 4 KiB chunks, 8 blocks, without
`fence.proxy.async.global`. The SASS confirms the only difference: `tma` has `FENCE.VIEW.ASYNC.G` after the acquire,
`tma_nofence` has none.

**Result.** `tma_nofence` read **fresh**. `tma` read fresh, and `nc_control` stale, in the same run.

**Conclusion.** The fresh check cannot detect a missing proxy fence on this path, so it does not certify the fence
safety of TMA or of LDGSTS. Both bypass L1, and the check's only demonstrated sensitivity is to a stale L1 line. Their
visibility rests on the PTX rules (acquire, then `fence.proxy.async.global` before an async-proxy read), not on this
measurement. No verdict changes: TMA and LDGSTS were not adopted (Decision-table rows above).

### 5c. Write-combined source slab (2026-09-27, commit 561715b4c9)

**Question.** On divix01's Sky Lake-E root port, does a source slab allocated with
`cudaHostAlloc(Mapped | WriteCombined)` raise SM or copy-engine H2D above the ordinary pinned slab? The GPU's reads of
it may skip the CPU snoop.

**Probe** (`wc_bench.py`, `wc_report.py`). Three source slabs with the sweep's rows and segments:

- `pinned`: `allocate_host_slab`, the sweep's.
- `hostalloc`: `cudaHostAlloc(Mapped)`. This control has the same allocator without WC, so any WC effect is not an
  allocator effect.
- `wc`: `cudaHostAlloc(Mapped | WriteCombined)`.

Both `cudaHostAlloc` slabs were allocated and written with the thread on the GPU's NUMA node 0 (cores 36-53). Each
shape ran once per slab per round, slabs interleaved, 3 rounds x 30 reps; the median over rounds is shown. The fresh
check ran against a WC source, with the host's rewrite fenced by `sfence` before the flag release (now in
`mech_fresh` for every method). The protocol words stayed in ordinary pinned memory. `WC_EXIT=0` and `REPORT_EXIT=0`;
no cell was above the ceiling.

```
| method | grid | a | pinned GB/s | hostalloc GB/s | WC GB/s | WC / pinned | hostalloc / pinned |
|---|---:|---:|---:|---:|---:|---:|---:|
| ce_batch | 0 | 4 | 13.692 | 13.688 | 13.701 | 1.001 | 1.000 |
| ce_each | 0 | 4 | 13.626 | 13.623 | 13.627 | 1.000 | 1.000 |
| sm_cv16 | 1 | 4 | 11.587 | 11.536 | 11.497 | 0.992 | 0.996 |
| sm_cv16 | 8 | 4 | 12.297 | 12.294 | 12.297 | 1.000 | 1.000 |
fresh sm_cv16@wc: True
fresh nc_control@wc: False
host pinned: write 4.03 GB/s, read 7.258 GB/s
host hostalloc: write 4.04 GB/s, read 7.125 GB/s
host wc: write 10.65 GB/s, read 0.048 GB/s
```

Per-round GB/s:

| shape | pinned | hostalloc | WC |
|---|---|---|---|
| `sm_cv16` G8 U4 (plateau) | 12.295 / 12.299 / 12.297 | 12.294 / 12.293 / 12.295 | 12.296 / 12.297 / 12.301 |
| `sm_cv16` G1 U4 (16 KiB) | 11.598 / 11.558 / 11.587 | 11.539 / 11.506 / 11.536 | 11.497 / 11.509 / 11.493 |
| `ce_each` 4 rows | 13.628 / 13.625 / 13.626 | 13.625 / 13.623 / 13.619 | 13.627 / 13.629 / 13.616 |
| `ce_batch` 4 rows | 13.686 / 13.694 / 13.692 | 13.688 / 13.687 / 13.688 | 13.706 / 13.701 / 13.693 |

**Result: WC does not raise H2D on divix01.**

- **Plateau and copy engine:** WC / pinned is 1.000 (SM plateau), 1.000 (`ce_each`) and 1.001 (`ce_batch`). Every
  difference is inside the round-to-round spread.
- **16 KiB, latency-bound:** WC is consistently about 0.8% slower (0.992). Its rounds, 11.49-11.51, do not overlap
  pinned's, 11.56-11.60. `hostalloc` is about 0.4% slower too, so roughly half of that gap comes from the allocator, not
  WC.
- **The snoop is not what limits these reads.** Both ceilings, 12.31 SM and 13.69 copy engine, stay where they were,
  consistent with 5a putting the limit in the link's request/completion efficiency.

**Fresh check against a WC source:** `sm_cv16@wc` fresh (true); `nc_control@wc` stale (false). So the check can still
see staleness on WC memory, and the fenced host rewrite is visible to `.cv` reads.

**Host side** (256 MiB memset + sfence; 32 MiB read):

| slab | CPU write GB/s | CPU read GB/s |
|---|---:|---:|
| pinned | 4.03 | 7.26 |
| hostalloc | 4.04 | 7.13 |
| WC | 10.65 | 0.048 |

CPU writes to WC are 2.6x faster, and CPU reads of it are about 150x slower.

**WC is not cleared for production, and there is no reason to adopt it.** The NVMe-write check (5d) then
tested the DDIO concern directly: no stale or wrong word in 2,100 `O_DIRECT` trials across pinned, `cudaHostAlloc`
and WC slabs, read by `.cv`, weak-after-acquire and copy-engine reads. That is evidence, not proof, and it is unknown
whether the WC reads were issued no-snoop at all (5d). WC also buys nothing on the GPU side on divix01, and it makes
any CPU read of the slab about 150x slower.

### 5d. NVMe O_DIRECT writes into a WC slab, then GPU reads (2026-09-27, commit 93e89cad13)

**Question.** Production fills the slabs with NVMe `O_DIRECT` reads. On Sky Lake-E, DDIO can leave those inbound
writes in the LLC. Does a GPU read of a write-combined slab, which may be issued no-snoop, then return stale DRAM
contents?

**Procedure** (`nvme_wc_check.py`, `nvme_report.py`). Run under `rowimg-disk.lock`, then the exclusive `cc-gpu.lock`,
with `TMPDIR` on NVMe. Source: the row images on the mirror root `/mnt/nvme0/dsv41_flash/exl3_row_images/`
(`layer-000..039.rows`), opened read-only. Each trial, on one slab:

1. The CPU writes pattern A into every 8 B word of the slab, then sfences. A is a per-trial counter, mixed and XORed
   with a constant; it occurred in no reference region (0 `pattern_in_file` records in 2,100 trials).
2. A kernel is launched that waits on a host flag. Its half-grids read the slab: 8 blocks with plain weak loads after
   the flag's acquire, 8 with `.cv` loads, S's pattern.
3. `pread` with `O_DIRECT` fills the slab from a 4 KiB-aligned region of a row image.
4. As `pread` returns, the host sfences and releases the flag, and at once issues `cudaMemcpyAsync` (the copy
   engine) of the slab on a stream of its own. So all three reads start together, with no CPU touch of the slab
   after the DMA.
5. On the GPU, each device copy is compared 8 B word by word with a reference of the same file region. The reference
   was loaded once through buffered I/O into ordinary memory and moved to the GPU; it was never read back from a slab.
   - A word is **stale** if it still equals A.
   - A word is **wrong** if it differs from the file at all.

Slabs, all page-aligned and interleaved per region, with the slab that reads the region first rotated each time:

- `pinned`: `allocate_host_slab`, NUMA node 0.
- `hostalloc`: `cudaHostAlloc(Mapped)`.
- `wc`: `cudaHostAlloc(Mapped | WriteCombined)`.

Both `cudaHostAlloc` slabs were allocated on the GPU's node 0.

Sizes and trials per slab: 64 KiB x 300, 1,662,976 B (one piece, 406 pages) x 300, 13,312,000 B (one row, 3,250
pages) x 100. Every trial read a distinct region of its size, rotating over the 40 layer files.

- **Row size ran 100 trials, not a few hundred.** That keeps the disk volume modest: about 5.5 GB of `O_DIRECT` plus
  1.9 GB of references. DDIO keeps only a few MB of LLC, so the 64 KiB and 1.66 MB sizes are the more sensitive ones.
- **Page cache.** mincore before and after: 0 of 4,800 and 0 of 121,800 pages resident at the two small sizes, and
  681 of 325,000 (0.2%) at the row size. The same 681 were resident before and after, so the trials did not load
  them. `O_DIRECT` reads the device regardless of cached clean pages.

`NVME_EXIT=0`, `REPORT_EXIT=0`. Result:

```
| slab | method | size (B) | trials | stale words | wrong words | trials with any |
|---|---|---:|---:|---:|---:|---:|
| hostalloc | ce | 65536 | 300 | 0 | 0 | 0 |
| hostalloc | sm_cv16 | 65536 | 300 | 0 | 0 | 0 |
| hostalloc | weak | 65536 | 300 | 0 | 0 | 0 |
| pinned | ce | 65536 | 300 | 0 | 0 | 0 |
| pinned | sm_cv16 | 65536 | 300 | 0 | 0 | 0 |
| pinned | weak | 65536 | 300 | 0 | 0 | 0 |
| wc | ce | 65536 | 300 | 0 | 0 | 0 |
| wc | sm_cv16 | 65536 | 300 | 0 | 0 | 0 |
| wc | weak | 65536 | 300 | 0 | 0 | 0 |
| hostalloc | ce | 1662976 | 300 | 0 | 0 | 0 |
| hostalloc | sm_cv16 | 1662976 | 300 | 0 | 0 | 0 |
| hostalloc | weak | 1662976 | 300 | 0 | 0 | 0 |
| pinned | ce | 1662976 | 300 | 0 | 0 | 0 |
| pinned | sm_cv16 | 1662976 | 300 | 0 | 0 | 0 |
| pinned | weak | 1662976 | 300 | 0 | 0 | 0 |
| wc | ce | 1662976 | 300 | 0 | 0 | 0 |
| wc | sm_cv16 | 1662976 | 300 | 0 | 0 | 0 |
| wc | weak | 1662976 | 300 | 0 | 0 | 0 |
| hostalloc | ce | 13312000 | 100 | 0 | 0 | 0 |
| hostalloc | sm_cv16 | 13312000 | 100 | 0 | 0 | 0 |
| hostalloc | weak | 13312000 | 100 | 0 | 0 | 0 |
| pinned | ce | 13312000 | 100 | 0 | 0 | 0 |
| pinned | sm_cv16 | 13312000 | 100 | 0 | 0 | 0 |
| pinned | weak | 13312000 | 100 | 0 | 0 | 0 |
| wc | ce | 13312000 | 100 | 0 | 0 | 0 |
| wc | sm_cv16 | 13312000 | 100 | 0 | 0 | 0 |
| wc | weak | 13312000 | 100 | 0 | 0 | 0 |
```

Median `pread` time (O_DIRECT, queue depth 1) is the same on every slab:

| size | pinned | hostalloc | WC |
|---:|---:|---:|---:|
| 64 KiB | 42.7 us | 42.9 us | 45.3 us |
| 1.66 MB | 646 us | 655 us | 665 us |
| 13.3 MB | 4.02 ms | 4.00 ms | 4.01 ms |

**Result: zero stale and zero wrong words in every cell.** That covers 3 slabs x 3 read methods x 3 sizes, 2,100
trials and 6,300 GPU reads.

**What this does and does not show.**

- **A zero is evidence, not proof.** Staleness from a no-snoop read racing a DDIO write is timing-dependent, and the
  reads here start about as early after the DMA as the host can arrange. A rare race could still exist below this
  sample's resolution.
- **Whether the WC slab's reads were actually no-snoop is unknown.**
  - The driver exposes no no-snoop setting: `/proc/driver/nvidia/params` has only `EnablePCIERelaxedOrderingMode: 0`.
  - No TLP capture was taken.
  - The GPU's DevCtl reads `RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+` (the user's `lspci -vvv`), so the GPU
    is *allowed* to set the no-snoop attribute, but whether the WC reads actually did remains unknown.

  If the reads were snooped, this check could not have failed. Then the zero says the driver does not use no-snoop
  for these reads, not that no-snoop would be safe.
- **Scope: this is a safety record, not a path to adoption.** WC gave no H2D speedup (5c), so nothing here argues for
  using it.
