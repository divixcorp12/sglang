# Project notes

To use nsys refer to '/opt/nvidia/nsight-systems/2026.5.1/skills/nsight-systems/SKILL.md'

## Nsight Systems traces

- PCIe RX/TX throughput comes from GPU metrics sampling (`--gpu-metrics-set=gb20x`). divix01's driver keeps the counters admin-only (`RmProfilingAdminOnly: 1`; as your user nsys reports "Insufficient privilege"), so `run_arm.sh` (`NSYS_GPU_METRICS=1`, the default) runs a second, metrics-only root session through the NOPASSWD wrapper `sudo -n /usr/local/sbin/nsys-profile` over the same window, written to `<report>-pcie.nsys-rep`, and refuses a traced arm when that path cannot read the counters. Metrics-only capture needs no server restart, so it also works against production. Its scratch lands in `/tmp/nsys-root` on the root volume, because sudo drops `NSYS_TMPDIR`; keep root captures short. "% of peak" is not yet calibrated to GB/s on this Gen3 x16 slot.
- Use `--cuda-graph-trace=graph` for decode traces, not `node`. Node mode makes `cudaGraphLaunch` cost ~0.77 µs per traced graph node (8,153 nodes → ~6 ms of host time per decode step), which shows up as a fake ~6 ms GPU-idle gap at the end of every step and inflates traced ms/token by ~7 ms (traced 74.1 vs untraced 66.8 ms/token). Use `node` only when a question needs per-kernel attribution inside the graph, and do not read step-tail idle or ms/token from such a trace. Evidence: `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/step-tail/`.
- The converse: **a graph-mode trace's kernel table leaves out the graph body.** Kernel summaries from it cover only eager work (prefill, sampling), so never rank kernels from one. Check first: if no kernel row has a nonzero `graphId` and summed kernel time is far below wall time, you are looking at prefill. Evidence: `MOE_EXPERT_TRANSFER.md`, "Nsight decode trace 2026-09-18".
- **A graph-mode report's `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` table can describe the warm-up, not the capture.** In `engram-on-trace-20260922-221602`, all 1,298 rows carry negative timestamps and correlation IDs strictly below the captured window's minimum: they are flushed pre-capture events on a different time base. Read naively they give a confident per-graph-execution average that describes nothing in the traced region. Check before using the table: `MIN(start)` against the other tables' `MIN(start)`, and whether its correlation IDs overlap `CUPTI_ACTIVITY_KIND_RUNTIME`'s. Summed graph time exceeding wall time is the tell.
- **`cudaMemcpyAsync` host time is not transfer time, and in this workload it is mostly not even a transfer.** Join each call to its GPU-side record by `correlationId` and read `bytes` before concluding anything about bandwidth: the same trace's biggest memcpy costs are 31-byte and 2 KB device-to-host readbacks (`.item()`) that block the host 841 us and 145 ms respectively while moving nothing. Evidence: `DSV41_REFERENCE.md` section 21.
- Bound every trace analysis's memory. On the laptop, `/tmp` is a RAM-backed tmpfs, and the Nsight skill writes its Parquet/DuckDB cache to `/tmp/nvidia/nsight_systems/nsys-skill-cache`. Loading a 250 MB node-mode decode report there out-of-memoried the laptop and crashed VS Code.
  - Analyse traces on divix01 where they live, under `taskset -c 0-63`. Copy a report to the laptop only when asked, and then only to a disk path, never `/tmp` or the scratchpad.
  - Cap memory for any local run: `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 <cmd>`. For DuckDB, also set `SET memory_limit='4GB'; SET threads=4`.
  - Narrow the data before loading it. Use `--filter-time`/`--filter-nvtx` or a short capture (`--duration`), and bound queries (`LIMIT`, pre-filtered CTEs).
  - Delete the skill cache and any copied report when done.
  - **On divix01, set `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp` for any full-length traced arm.** Its `/tmp` is on the root xfs volume at ~88% full; nsys wants 200 MiB of scratch and a full 8-session capture exhausts it. The failure is not a clean error -- nsys warns, the server is then killed mid-run, the benchmark driver reports `RemoteDisconnected`, and the report lands at ~361 KB. A short 2-session capture fits and hides the problem. Evidence: `DSV41_REFERENCE.md` section 22.

## GPU microbenchmarks on the RTX 5090

- **Size the working set past L2 (96 MiB) or you are measuring L2, not HBM.** A
  benchmark that reuses one tensor across repetitions stays resident and reports
  bandwidth the real workload will never see. This produced two recorded
  constants that understate production cost by 1.5x and 2.3x
  (`expert_residency_gpu.py`, the `0.007`/`0.054` ms/row pair); the fix was to
  spread the same work over 48 separate layer tensors for a 4.9 GiB footprint.
- **Sanity-check the implied bandwidth against the card's spec.** An impossible
  number is the cheapest detector available — a 3.1 TB/s result is what exposed
  the cache-resident benchmark above. HBM and PCIe ceilings both work for this.
- PCIe transfers cannot be flattered by L2, so H2D figures are unaffected; this
  applies to device-to-device work.

## magic-trace / DSpark CPU traces

- The portable skill lives in `tools/skills/magic-trace/` and is installed on
  the laptop as `~/.codex/skills/magic-trace/`. Invoke it with “magic-trace
  <binary>” or `$magic-trace`. Keep download/setup, Intel PT/perf compatibility,
  ELF/PIE address resolution, validated snapshots, laptop copying and viewer
  instructions in the skill; the points below describe this repository.
- Start from the normal optimized JIT build. The exported
  `sglang_draft_delay_trigger` uses `noinline` plus volatile asm so its real call
  survives optimization. The original call sites lived inside
  `if constexpr (Build::kMetrics)`; disabling metrics discarded those branches.
  That is compile-time elimination, not an inlining bug. A counters-off capture
  needs an independent trigger build switch and reachable call site; preserving
  the function symbol alone does not establish that the engine can call it.
- `benchmarks/dsv41_baseline/run_omp_serving_capture.py` is the existing serving
  capture driver. `--magic-trace PATH` selects Intel PT, and
  `--draft-arrival-trigger-us`, `--draft-pending-trigger-us`, and
  `--draft-forward-trigger-us` select threshold reasons 2, 0 and 1 respectively.
  The original metrics mode uses
  `SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX` and a bounded mapped trace gate. Wait
  for the collector's `[ Attached.` before opening that gate. A
  `--magic-health-diagnostic` capture starts at HTTP health and is diagnostic
  only; it does not establish steady-state throughput or bypass benchmark gates.
- For detailed-counters-off comparisons, the driver's `--no-instrumentation`
  selects `SGLANG_DRAFT_DELAY_TRIGGER_ONLY=1` and a separately cached
  `prod_trigger` host DSO with metrics false. It retains
  `SGLANG_DRAFT_FORWARD_TRIGGER_US` and the
  `SGLANG_CPU_EXPERT_TRACE_GATE` admission gate; arrival/pending thresholds are
  unavailable in this mode. The trigger reuses production forward start/end
  clocks, checks the gate once per completed forward and fires once, with no
  extra clock reads or event buffers. Optional
  `SGLANG_DRAFT_DELAY_TRIGGER_REPORT_PREFIX` saves one trigger result at
  shutdown. Label this trigger-only, detailed-counters-off capture precisely.
- `magic-work/hits.sexp` establishes the native trigger entry/TID/IP, but its
  `passed_timestamp` and `passed_val` fields use magic-trace v1.2.4's OCaml
  register decoding, not our C++ `(seq, elapsed_ns, reason)` signature. In the
  counters-off capture, the native report's 5,044,085 ns elapsed appeared as
  2,522,042 in `passed_val`; the passed timestamp was meaningless. Read the
  `SGLANG_DRAFT_DELAY_TRIGGER_REPORT_PREFIX` shutdown report (or detailed app
  marker) for sequence, elapsed time, reason and threshold. The portable skill's
  trigger reference links the verified upstream decoder implementation.
- Resolve the active instrumented **host** JIT DSO from the live engine's
  ownership, build identity and mappings. Multiple loaded host libraries can
  export the same trigger, including unused variants. Do not select the last
  matching mapping. The trigger's printed attach address must match the live
  address after PIE adjustment. OpenMP workers inherit `exl3-cpu-exp0`/`1`
  names: a matching `comm` alone does not identify the engine leader. Use the
  runtime manifest, owning scheduler and affinity/creation model to select its
  actual TID. Save this resolution with the capture.
- GPU publication uses `%globaltimer`; CPU markers use `CLOCK_MONOTONIC`.
  Correlate them with the saved `draft-clock*.json` anchors and retain their
  uncertainty/drift limits. `draft_start` follows request reading/compaction;
  use `draft_observed`, `draft_selected`, `draft_record_ready` and
  `draft_payload_ready` to inspect preceding delay. Do not interpret the GPU
  publication timestamp as the exact release/head-store instant.
- The divix01 tool tested here is
  `/data/models/slang/nvfp4-work/tools/magic-trace-v1.2.4`. Its perf 6.12 decoder
  needs the narrow `tr strt jmp` adapter:
  `MAGIC_TRACE_PERF_PATH=$PWD/benchmarks/dsv41_baseline/magic_trace_perf_compat.py`.
  The portable skill has its own independent adapter. Keep raw `perf.data`.
  A tiny FXT/exit 0/“Snapshot taken” can still mean no events and no saved PT
  payload; validate both the timeline and AUXTRACE records before delivery.
- Reference model capture (2026-10-07): laptop
  `~/Downloads/dsv41-magic-trace-20261007/model-capture/draft-delay.fxt.gz`,
  SHA256 `ed80ce216b381489348cbda09553033a3faf9163e0ded021010e118647ebe319`,
  91,928 events. It triggered after a 5.04 ms draft forward and has five PT
  overflows, so gaps are not evidence of continuous execution. Siblings named
  `failed-worker-capture` and `failed-trigger-capture` are rejected attempts,
  not usable timelines. Seeing `keep_warm_either` frames establishes control
  flow, not the cause of a memory or scheduling stall.
- Follow `.claude/rules/divix01-run-protocol.md`: commit/push/pull code into a
  private worktree, cap CPU jobs to cores 0–63 with `OMP_NUM_THREADS`, and use
  the disk/GPU locks for serving. Write artifacts to disk, preserve failed
  attempts separately, and checksum completed laptop copies. Shut down normally
  when bounded application buffers need flushing.
