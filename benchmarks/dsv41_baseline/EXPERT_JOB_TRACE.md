# CPU expert and copy completion tracing

Set `SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX=/absolute/disk/path/events` on a diagnostic server launch. The Python host selector loads `InstrBuild`; `ProdBuild` compiles the event buffers, observation state, branches and extra clock reads out. Existing CPU aggregate timing remains unchanged.

Each CPU engine and copy engine reserves 131,072 event slots once. Events contain host `CLOCK_MONOTONIC` timestamps and are written to `<prefix>.<pid>.<thread-name>.<instance>.jsonl` after engine shutdown. The parent directory must already exist. Normal shutdown is required; an aborted process cannot flush its buffers. A footer reports overflow, and the analysis command refuses incomplete or dropped-event files.

`SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY` optionally changes the per-engine bound (1–1,048,576 events). The diagnostic driver uses 524,288 so resource snapshots during readiness warm-ups do not exhaust the timed request's buffer. This allocates about 32 MiB per engine; always inspect every overflow footer. Production does not read this setting or allocate a buffer.

The first line contains a monotonic/Unix epoch clock anchor for approximate alignment with Nsight session timestamps. Its two sequential clock reads have a small alignment uncertainty. CPU and copy attribution itself uses the shared monotonic clock.

```bash
python benchmarks/dsv41_baseline/expert_job_trace.py \
  '/absolute/disk/path/events.*.jsonl' --output findings.json
```

The command writes detailed per-record/per-group results and an instant-event Chrome/Perfetto timeline beside the JSON summary. To restrict attribution to a capture window, pass `--start-ns` and `--end-ns` in host monotonic nanoseconds. These filters select records by gate completion after joining their full dependency history.

CPU events identify target jobs by NUMA engine, layer row and sequence, with submit/start/end timestamps. `cpu_shape` records token rows, unique expert lanes and live token/expert routes; `cpu_submit.a` distinguishes hit part 0 from forced-miss part 1. Draft start/end events use the draft stage and channel epoch/sequence.

For copies, `gen` identifies the target record across both NUMA groups. `copy_submit.seq` is the CPU sequence range start; `.a` is the last forced-miss sequence, `.b` the forced-miss lane count, and `.c` the CPU-hit lane mask. `copy_issue.a` is DMA bytes, `.b` DMA lanes, and `.c` the completion token. `group_done` marks FIFO retirement; `gate_open` follows publication of combined completion for every required group.

`copy_dma_observed.a` is the timestamp before the last poll that returned pending, and `.ns` is the first observed done timestamp. Actual DMA completion lies in that interval. `copy_cpu_observed` is an independent observation of CPU completion. The observer checks every in-flight group, so FIFO retirement cannot hide a group that completed earlier. These are host observations, not GPU transfer timestamps.

For earlier captures missing the observer-done marker, retirement supplies a conservative upper bound with an unknown lower bound; `dma_bound_source` flags this fallback.

The classifier confirms CPU-last only when CPU end is later than the DMA upper bound, and DMA-last only when the last pending poll follows CPU end. It uses a 10 microsecond margin and leaves overlapping intervals ambiguous. Copy-only and CPU-only groups have separate classifications. Results do not include the earlier GPU-post-to-host-submit delay or GPU work after the gate.

Instrumented tracing adds polling and clock overhead. It measures dependency order and workload shape; use an untraced run to measure production throughput. Keep other benchmarks and builds idle during a capture and follow the disk/GPU lock order in `.claude/rules/divix01-run-protocol.md`.

## Diagnosing CPU pauses

Also set `SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE=1` to record `cpu_faults_start/end` and `cpu_switches_start/end` (and corresponding `draft_` events). Fault events carry cumulative minor faults, major faults, and Linux TID in `a/b/c`. Switch events carry cumulative voluntary switches, involuntary switches, and thread user+system CPU nanoseconds in `a/b/c`. Subtract each matching end/start pair. These counters cover the engine leader only; they do not sum its OpenMP workers. Resource syscalls compile out of production, are opt-in within the instrumented build, and add four events per job. Negative values mean collection failed. The extra events can overflow the bounded buffer on longer captures.

Run `stall_sampler.py --prefix <same-prefix> --output <samples.jsonl> --stop-file <stop-marker>` in parallel, pinned to a spare core outside the expert teams and IRQ cores. It selects the scheduler by the exact trace-prefix environment entry, then samples named EXL3 threads' faults, context switches, scheduler runtime/runqueue delay, state and wait channel every 50 ms, plus global/per-node memory counters every 250 ms. Touch the stop marker when the arm ends. The default hard bounds are 40 minutes and 128 MiB; its footer reports limits and inaccessible reads. A zero/empty wait channel is inconclusive, and waits shorter than the sampling interval may be missed.

For a short Nsight arm, `NSYS_SYSTEM_CPU=1` extends the existing root `-pcie.nsys-rep` companion with system-wide CPU sampling/scheduling and ftrace scheduler, user-fault, direct-reclaim and compaction events. It requires `NSYS_TRACE=1` and `NSYS_GPU_METRICS=1` plus the existing root Nsight wrapper. The normal main report can use `NSYS_SAMPLE=none NSYS_CPUCTXSW=none` to avoid duplicate CPU sampling. Verify ftrace event availability with a short capture first. This companion still contains PCIe metrics; it is no longer metrics-only in this mode. The root collector's scratch remains on the root volume, so keep the capture short and check disk space.

`run_stall_capture.py --no-nsys --reference <capture-command.json> --output <new-directory>` repeats the same serving recipe with just job resources and the bounded `/proc` sampler. It disables Nsight entirely, including CUDA/OSRT injection and root metrics collection. Use this to investigate profiler effects; instrumentation still has overhead, so it is not a production throughput baseline.

## Per-worker work and barriers

Add `--worker-phases` to that diagnostic command. It sets `SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX=<output>/worker-phases`, which builds the optimized EXL3 CPU extension with `EXL3_MOE_CPU_WORKER_TRACE=1` in a separate `worker_trace/` build directory. Job tracing must also be enabled for attribution. Normal builds instantiate the empty capture specialization: no worker timing state, resource syscalls or trace buffers on their forward path.

Each forward records worker-local phase start, work completion and barrier exit in `CLOCK_MONOTONIC`, with matching `CLOCK_THREAD_CPUTIME_ID` snapshots. IDs are 0 prepare gate/up, 1 gate/up matrix work, 2 activation/down preparation, 3 down matrix work, 5 accumulation (4 is a fused historical phase). Accumulation has no explicit phase barrier; the team join follows it. `team_begin` follows scratch preparation; each worker's `enter` precedes pinning/the initial barrier, and `ready` follows them. Frame times exclude route grouping before the plan. Work units in phases 1/3 are assigned output tiles multiplied by chunk token rows, not a measured instruction count. Worker fault and context-switch deltas span entry through the worker body and repeat on each phase row; count them once per worker/forward.

Only forwards exceeding `SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US` (default 6000) are retained. The default `SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY=65536` bounds phase records per calling thread; the driver uses 131072. A whole forward is admitted or dropped, with explicit footer counts. Buffers allocate once per calling thread and flush at thread exit to `<prefix>.<pid>.<leader-tid>.jsonl`; no allocation, formatting or IO occurs while retaining a forward. Up to 64 workers are supported. Join a frame to the enclosing CPU/draft job by its leader TID and monotonic interval. Always check footers and distinguish retained tails from all-job statistics.

For each phase compare latest worker work completion with barrier exits. A worker whose work wall time greatly exceeds CPU time was blocked or descheduled; CPU time tracking work wall time suggests active execution. Unequal weighted tile assignments can explain unequal work, but units do not model every kernel's cost. A large delay after every worker finished suggests barrier/wake-up overhead. These are measurements with observer overhead: thread-CPU-clock and resource syscalls themselves can perturb scheduling, and their sampling edges are sequential.

## Draft arrival and magic-trace

The instrumented host also records `draft_observed` (first pending head read, before target-queue selection),
`draft_selected` (serve entry), `draft_record_ready` (seqlock read/validation), and `draft_payload_ready`
(compacted routes and constructed forward call). Their row/gen/seq identify stage/epoch/request;
`draft_observed.a` is the raw GPU `%globaltimer` marker and `.b` the engine leader TID.
The GPU marker precedes record/head publication, so its interval includes release work and visibility.
It is compiled into the device post kernel only when the job trace prefix selects the diagnostic build.

`SGLANG_DRAFT_PENDING_TRIGGER_US` measures first observation to payload-ready; `SGLANG_DRAFT_FORWARD_TRIGGER_US`
measures kernel-forward wall time. `SGLANG_DRAFT_ARRIVAL_TRIGGER_US` uses the startup-correlated publication-to-first-observation lower estimate. Zero disables each threshold. The first admitted threshold crossing per engine
emits `draft_trigger` (a: reason 0 pending / 1 forward / 2 arrival, b: measured ns, c: threshold ns) and calls the exported
`sglang_draft_delay_trigger`. Its `noinline` attribute and volatile asm preserve an actual call in optimized builds.
Warm-up while the trace gate is closed cannot consume the trigger. Production host builds compile this path out.

`run_omp_serving_capture.py --magic-trace /absolute/path/magic-trace ...` replaces the scheduler-only Nsight
collector with a bounded Intel PT snapshot on the proven-owned node 0 engine leader. It resolves the trigger's
address in the loaded JIT DSO, waits for attachment before opening the timed gate, and retains raw perf data.
The initial thresholds are 500us arrival, 500us pending and 5000us forward. If no crossing occurs, the capture stops after
`--seconds`; this is a fallback, not evidence of a triggered stall. Keep the normal optimized compile settings first.

`analyze_draft_arrival.py <capture-directory>` checks completeness/overflow and joins the timings.
Initialization writes 64 host-bracketed GPU clock samples to `events.<pid>.draft-clock.json`.
The intersection of offset intervals gives correlation uncertainty assuming a stable nanosecond clock offset;
a startup anchor alone does not bound later drift. Report raw GPU and CPU clocks and that limitation.
The arrival trigger uses the startup offset and therefore requires the same drift checks as the analysis.
The capture harness records a second clock anchor after shutdown; reject precise GPU correlations if the offset intervals disagree. Intel PT explains native control flow, not GPU execution or DRAM stall causes.

On divix01, magic-trace v1.2.4 initially failed to parse perf 6.12's `tr strt jmp` flag.
Set `MAGIC_TRACE_PERF_PATH=$PWD/benchmarks/dsv41_baseline/magic_trace_perf_compat.py` to canonicalize only
that alias to `tr strt`, matching upstream master's decoding semantics. Recording uses `/usr/bin/perf`
unchanged; raw perf data is retained. This adapter is not a blanket parser-error filter.

If the existing 12-round throughput-stability gate fails, retain that failed arm. An explicit
`--magic-health-diagnostic` capture can instead attach after HTTP health and record a warm-up request.
It waits for the current request to complete after the profiler stops, then shuts down normally.
It writes `diagnostic.start` and `diagnostic-completion.json`, and produces no timed benchmark or PASS verdict.
Use this to investigate fluctuating execution; it cannot establish steady-state throughput.
